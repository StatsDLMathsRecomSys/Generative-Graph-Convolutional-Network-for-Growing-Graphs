from __future__ import print_function, division
import math
import numpy as np
import torch
from graph_data import preprocess_graph
from utils import sample_reconstruction

def weighted_cross_entropy_with_logits(logits, targets, pos_weight):
    """
    see: https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    """
    #logits = torch.clamp(logits, min=-10, max=10)

    x = logits
    z = targets
    l = 1 + (pos_weight - 1) * targets

    loss = (1 - z) * x + l * (torch.log(1 + torch.exp(-torch.abs(x))) + torch.clamp(-x, min=0))
    return loss

    # return targets * -torch.log(torch.sigmoid(logits)) * pos_weight + (1 - targets) * -torch.log(1 - torch.sigmoid(logits))

def KL_normal(z_mean_1, z_std_1, z_mean_2, z_std_2):

    kl = torch.log(z_std_2 / z_std_1) + ((z_std_1 ** 2) + (z_mean_1 - z_mean_2) ** 2) / (2 * z_std_2 ** 2) - 0.5
    return kl.sum(1).mean()
    #return torch.mean(kl)

def reconstruction_loss(adj, adj_h,mask=None, test=False, fixed_norm=None):
    if not test:
        norm = adj.shape[0] ** 2 / float((adj.shape[0] ** 2 - adj.sum()) * 2)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    else:
        norm = 1.0
        pos_weight = 1.0

    if fixed_norm is not None:
        norm = fixed_norm
        pos_weight = 1.0


    element_loss = weighted_cross_entropy_with_logits(adj_h, adj, pos_weight)
    if mask is not None:
        element_loss = element_loss[mask]
    neg_log_lik = norm * torch.mean(element_loss)
    return neg_log_lik

def vae_loss(z_mean, z_log_std, adj, fixed_norm=None):
    adj_h = sample_reconstruction(z_mean, z_log_std)
    neg_log_lik = reconstruction_loss(adj, adj_h, fixed_norm=fixed_norm)
    z_std = torch.exp(z_log_std)
    kl = KL_normal(z_mean, z_std, 0.0, 1.0)
    # kl = torch.mean(torch.log(1 / z_std) + (z_std ** 2 + (z_mean - 0) ** 2) * 0.5)
    return neg_log_lik + kl / z_mean.size()[0]


def r_vae_loss(z_mean_old, z_log_std_old, z_mean_new, z_log_std_new, adj, fixed_norm=None):

    z_mean = torch.cat((z_mean_old, z_mean_new))
    z_log_std = torch.cat((z_log_std_old, z_log_std_new))

    adj_h = sample_reconstruction(z_mean, z_log_std)
    loss = reconstruction_loss(adj, adj_h, fixed_norm=fixed_norm)
    z_std_new = torch.exp(z_log_std_new)
    kl = KL_normal(z_mean_new, z_std_new, 0.0, 1.0)
    # kl = torch.mean(torch.log(1 / z_std_new) + (z_std_new ** 2 + (z_mean_new - 0) ** 2) * 0.5)
    loss += kl * (z_mean_new.size()[0] / z_mean.size()[0] ** 2)

    return loss

def r_vae_loss_addon(last_z_mean, last_z_log_std, z_mean_old, z_log_std_old, z_mean_new, z_log_std_new, adj, fixed_norm=None):

    z_mean = torch.cat((z_mean_old, z_mean_new))
    z_log_std = torch.cat((z_log_std_old, z_log_std_new))

    adj_h = sample_reconstruction(z_mean, z_log_std)
    loss = reconstruction_loss(adj, adj_h, fixed_norm=fixed_norm)

    last_z_std = torch.exp(last_z_log_std)
    z_std_old = torch.exp(z_log_std_old)


    kl_last = KL_normal(z_mean_old, z_std_old, last_z_mean, last_z_std)
    #kl_last = torch.mean(torch.log(last_z_std / z_std_old) + (z_std_old ** 2 + (z_mean_old - last_z_mean) ** 2) * 0.5)
    kl_last *= (z_mean_old.size()[0] / z_mean.size()[0] ** 2)

    z_std_new = torch.exp(z_log_std_new)
    kl_new = KL_normal(z_mean_new, z_std_new, 0.0, 1.0)
    #kl_new = torch.mean(torch.log(1 / z_std_new) + (z_std_new ** 2 + (z_mean_new - 0) ** 2) * 0.5)
    kl_new *= (1.0 / z_mean.size()[0] ** 2)
    loss += kl_last
    loss += kl_new
    return loss

def recursive_loss(gcn_step, adj, feat, size_update, fixed_norm=1.2):
    num_step = int(math.ceil(1.0 * adj.size()[0] / size_update))

    # print("num step: {}".format(num_step))

    loss = torch.tensor([0.0])
    for step in range(num_step):

        if step == 0:
            adj_feed = torch.eye(size_update)
            feat_feed = feat[:size_update, :]
            z_mean, z_log_std = gcn_step(adj_feed, feat_feed)

            adj_truth = adj[0:size_update, 0:size_update]
            loss += vae_loss(z_mean, z_log_std, adj_truth, fixed_norm=fixed_norm)

            continue

        start_idx = step * size_update
        end_idx = min(adj.size()[0], start_idx + size_update)
        adj_feed = adj[:start_idx, :start_idx].numpy()
        adj_feed_norm = preprocess_graph(adj_feed)
        adj_feed_norm = torch.from_numpy(adj_feed_norm)

        feat_feed = feat[:start_idx, :]
        fead_new = feat[start_idx:end_idx, :]
        z_mean_old, z_log_std_old, z_mean_new, z_log_std_new = gcn_step(adj_feed_norm, feat_feed, fead_new)
        adj_truth = adj[:end_idx, :end_idx]
        curr_loss = r_vae_loss(z_mean_old, z_log_std_old, z_mean_new, z_log_std_new, adj_truth, fixed_norm=fixed_norm)

        loss += curr_loss * end_idx ** 2
    return loss / num_step


def recursive_loss_with_noise(gcn_step, adj, feat, size_update, fixed_norm=1.2):
    num_step = int(math.ceil(1.0 * adj.size()[0] / size_update))

    #print("num step: {}".format(num_step))

    last_z_mean = None
    last_z_log_std = None
    for step in range(num_step):

        if step == 0:
            #adj_feed = torch.eye(size_update)
            adj_feed = adj[:size_update, :size_update]
            feat_feed = feat[:size_update, :]
            z_mean, z_log_std = gcn_step(adj_feed, feat_feed)

            adj_truth = adj[0:size_update, 0:size_update]
            loss = vae_loss(z_mean, z_log_std, adj_truth, fixed_norm=fixed_norm)
            last_z_mean, last_z_log_std = z_mean, z_log_std
            continue

        start_idx = step * size_update
        end_idx = min(adj.size()[0], start_idx + size_update)
        adj_feed = adj[:start_idx, :start_idx]

        feat_feed = feat[:start_idx, :]
        fead_new = feat[start_idx:end_idx, :]
        z_mean_old, z_log_std_old, z_mean_new, z_log_std_new = gcn_step(adj_feed, feat_feed, fead_new)
        adj_truth = adj[:end_idx, :end_idx]

        curr_loss = r_vae_loss_addon(last_z_mean, last_z_log_std, z_mean_old, z_log_std_old, z_mean_new, z_log_std_new, adj_truth, fixed_norm=fixed_norm)
        loss += curr_loss * end_idx ** 2

        # update hidden latent spaces
        #last_z_mean = torch.cat((z_mean_old, z_mean_new))
        # last_z_log_std = torch.cat((z_log_std_old, z_log_std_new))
        adj_feed = adj[:end_idx, :end_idx]
        feat_feed = feat[:end_idx, :]
        last_z_mean, last_z_log_std = gcn_step(adj_feed, feat_feed)


    return loss / num_step


def recursive_loss_with_noise_supervised(gcn_step, adj, label, feat, size_update, fixed_norm=1.2):
    num_step = int(math.ceil(adj.size()[0] / size_update))

    # print("num step: {}".format(num_step))

    last_z_mean = None
    last_z_log_std = None
    for step in range(num_step - 1):

        if step == 0:
            adj_feed = torch.eye(size_update)
            feat_feed = feat[:size_update, :]
            z_mean, z_log_std = gcn_step(adj_feed, feat_feed)

            label_truth = label[0:size_update, 0:size_update]
            loss = vae_loss(z_mean, z_log_std, label_truth, fixed_norm=fixed_norm)
            last_z_mean, last_z_log_std = z_mean, z_log_std
            continue

        start_idx = step * size_update
        end_idx = min(adj.size()[0], start_idx + size_update)
        adj_feed = adj[:start_idx, :start_idx]

        feat_feed = feat[:start_idx, :]
        fead_new = feat[start_idx:end_idx, :]
        z_mean_old, z_log_std_old, z_mean_new, z_log_std_new = gcn_step(adj_feed, feat_feed, fead_new)
        label_truth = label[:end_idx, :end_idx]

        curr_loss = r_vae_loss_addon(last_z_mean, last_z_log_std, z_mean_old, z_log_std_old, z_mean_new, z_log_std_new, label_truth, fixed_norm=fixed_norm)
        loss += curr_loss * (end_idx + 1) ** 2

        # update hidden latent spaces
        last_z_mean = torch.cat((z_mean_old, z_mean_new))
        last_z_log_std = torch.cat((z_log_std_old, z_log_std_new))
    return loss / num_step

def rgcn_loss(z_prior, z_post, adj):
    kl = torch.tensor(0.0)
    num_const1 = 0
    num_const2 = 0
    for i in range(len(z_post)):
        num_const2 += z_post[i][0].size(0) ** 2
        num_const1 += z_post[i][0].size(0)
    for i in range(len(z_post)):
        z_post_mean = z_post[i][0]
        z_post_std = torch.exp(z_post[i][1])
        z_prior_mean = z_prior[i][0]
        z_prior_std = torch.exp(z_prior[i][1])
        curr_kl = KL_normal(z_post_mean, z_post_std, z_prior_mean, z_prior_std) * z_post_mean.size(0) / num_const1
        #print(curr_kl)
        kl += curr_kl
    neg_log_lik = torch.tensor(0.0)
    for z_mean, z_log_std in z_post:
        adj_h = sample_reconstruction(z_mean, z_log_std)
        idx = z_mean.size(0)
        neg_log_lik += reconstruction_loss(adj[:idx, :idx], adj_h) * idx ** 2 / num_const2
    print(neg_log_lik)
    print(kl)
    return neg_log_lik + kl

def rgcn_loss1(z_prior, z_post, adj):
    kl = torch.tensor(0.0)
    num_const = 0
    for i in range(len(z_post)):
        num_const += z_post[i][0].size(0) ** 2
    for i in range(len(z_post)):
        z_post_mean = z_post[i][0]
        z_post_std = torch.exp(z_post[i][1])
        z_prior_mean = z_prior[i][0]
        z_prior_std = torch.exp(z_prior[i][1])
        curr_kl = KL_normal(z_post_mean, z_post_std, z_prior_mean, z_prior_std) * z_post_mean.size(0) / num_const
        #print(curr_kl)
        kl += curr_kl
    neg_log_lik = torch.tensor(0.0)
    z_mean = z_post[-1][0]
    z_log_std = z_post[-1][1]
    adj_h = sample_reconstruction(z_mean, z_log_std)
    print(adj_h.shape)
    print(adj.shape)
    neg_log_lik = reconstruction_loss(adj[:adj_h.shape[0],:adj_h.shape[0]], adj_h)
    print(neg_log_lik)
    print(kl)
    return neg_log_lik + kl
