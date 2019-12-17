#!/Users/d0x00ar/anaconda3/bin/python3.6
import math
import networkx as nx
import numpy as np
import torch
import os
import scipy.sparse as sp
from torch.utils.data import Dataset

from layers import RecursiveGraphConvolutionStep, RecursiveGraphConvolutionStepAddOn, GraphVae, MLP
from graph_data import generate_random_graph_data, GraphSequenceBfsRandSampler, preprocess_graph
from loss import recursive_loss, reconstruction_loss, vae_loss, recursive_loss_with_noise_supervised
from utils import link_split_mask, sample_reconstruction, get_roc_auc_score, get_average_precision_score, get_equal_mask

#os.chdir("/Users/d0x00ar/Documents/GitHub/R-GraphVAE/src")
import logging

#meta
size_update = 20
hidden_dim = 64
out_dim = 32
cite_data = 'citeseer'
random_h = True
seed = None
unseen = True
refit = False
permute = False
norm=None
special = 'GAE'
seed = 888
num_permutation = 400
weight_decay = 0
dropout=0.0

head_info = '_'.join(['equal_size_cite', special, cite_data,
                    'size', str(size_update),
                    'hidden', str(hidden_dim),
                    'out', str(out_dim),
                    'random_h', str(random_h),
                    'fix', str(seed is not None),
                    'unseen', str(unseen),
                    'refit', str(refit),
                    'norm', str(norm),
                    'permute', str(permute),
                    'seed', str(seed)])

filename = 'train_original_gae_64_32_citeseer_nodropout'
filename = '../data/important_results/' + filename


logging.basicConfig(level=logging.DEBUG, filename=filename,
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.info('This use the sample implementation.')
logging.info(head_info)

if seed is not None:
    np.random.seed(seed)

###############  Prepare input data ###############
def read_citation_dat(dataset, with_test=False, permute=False, test_ratio=0.3):
    '''
    dataset: {'cora', 'citeseer', 'pubmed'}
    '''

    feat_fname = '../data/' + dataset + '_features.npz'
    adj_fname = '../data/' + dataset + '_graph.npz'
    features = sp.load_npz(feat_fname)
    adj_orig = sp.load_npz(adj_fname)

    adj_orig = adj_orig + sp.eye(adj_orig.shape[0])

    X_select = features.todense().astype(np.float32)
    adj_select = adj_orig.todense().astype(np.float32)

    if permute:
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_select = adj_select[np.ix_(x_idx, x_idx)]
        X_select = X_select[x_idx, :]

    if with_test:
        cut_idx = int(adj_select.shape[0] * (1 - test_ratio))

        adj_train = adj_select[:cut_idx, :cut_idx]
        X_train = X_select[:cut_idx, :]
        return adj_train, X_train, adj_select, X_select
    else:
        return adj_select, X_select



g_adj, X, g_adj_all, X_all = read_citation_dat(cite_data, with_test=True)

features_dim = X.shape[1]



params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 2}


############### Init models ###############
gcn_vae = GraphVae(features_dim, hidden_dim, out_dim, dropout=dropout)
mlp = MLP(features_dim, hidden_dim, out_dim)

optimizer_vae = torch.optim.Adam(gcn_vae.parameters(), lr=1e-2)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-2)
train_loss = 0

cache = None
################ training loop #####################
adj = torch.from_numpy(g_adj)
feat = torch.from_numpy(X)
for batch_idx in range(num_permutation):

    if adj.size()[0] <= size_update:
        print("sample size {} too small, skipped!".format(adj.size()[0]))
        continue

    # train GCN
    optimizer_vae.zero_grad()
    gcn_vae.train()
    adj_vae_norm = torch.from_numpy(preprocess_graph(adj.numpy()))
    z_mean, z_log_std = gcn_vae(adj_vae_norm, feat)
    vae_train_loss = vae_loss(z_mean, z_log_std, adj, norm)
    vae_train_loss.backward()
    optimizer_vae.step()

    # train mlp
    optimizer_mlp.zero_grad()
    mlp.train()
    z_mean, z_log_std = mlp(feat)
    mlp_train_loss = vae_loss(z_mean, z_log_std, adj)
    mlp_train_loss.backward()
    optimizer_mlp.step()

    if batch_idx % 10 == 0:

        info = 'GCN [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx, num_permutation,
                100. * batch_idx / num_permutation,
                vae_train_loss.item())
        print(info)
        logging.info(info)

        info = 'MLP [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx, num_permutation,
                100. * batch_idx / num_permutation,
                mlp_train_loss.item())
        print(info)
        logging.info(info)

        with torch.no_grad():

            adj_feed_norm = torch.from_numpy(preprocess_graph(g_adj))
            adj_all = torch.from_numpy(g_adj_all)
            feat = torch.from_numpy(X)
            feat_all = torch.from_numpy(X_all)

            mask = torch.ones_like(adj_all).type(torch.ByteTensor)
            mask[:feat.size()[0], :feat.size()[0]] = 0

            diag_mask = torch.eye(mask.size(0)).type(torch.ByteTensor)
            mask = mask * (1 - diag_mask)

            mask = get_equal_mask(adj_all, mask, thresh=0.8)

            # test original gcn
            gcn_vae.eval()
            adj_vae_norm = torch.eye(feat_all.size()[0])
            adj_vae_norm[:feat.size()[0], :feat.size()[0]] = adj_feed_norm
            z_mean, z_log_std = gcn_vae(adj_vae_norm, feat_all)
            adj_h = sample_reconstruction(z_mean, z_log_std)
            test_loss = reconstruction_loss(adj_all, adj_h, mask, test=True)
            auc_gcn = get_roc_auc_score(adj_all, adj_h, mask)
            ap_gcn = get_average_precision_score(adj_all, adj_h, mask)

            info = 'Original GCN test loss: {:.6f}'.format(test_loss)
            print(info)
            logging.info(info)


            # test on mlp
            mlp.eval()
            z_mean, z_log_std = mlp(feat_all)
            adj_h = sample_reconstruction(z_mean, z_log_std)
            test_loss = reconstruction_loss(adj_all, adj_h, mask, test=True)
            auc_mlp = get_roc_auc_score(adj_all, adj_h, mask)
            ap_mlp = get_average_precision_score(adj_all, adj_h, mask)
            info = 'MLP test loss: {:.6f}'.format(test_loss)
            print(info)
            logging.info(info)

            ###### refit model ######
            # test original gcn
            gcn_vae.eval()
            adj_vae_norm = torch.eye(feat_all.size()[0])
            adj_vae_norm[:feat.size()[0], :feat.size()[0]] = adj_feed_norm
            z_mean, z_log_std = gcn_vae(adj_vae_norm, feat_all)
            adj_h = sample_reconstruction(z_mean, z_log_std)
            # ri-fit
            adj_fake = (adj_h.sigmoid() > 0.8).type(torch.FloatTensor)
            adj_fake[:feat.size(0), :feat.size(0)] = torch.from_numpy(g_adj)
            adj_fake = adj_fake * (1 - torch.eye(adj_fake.size(0))) + torch.eye(adj_fake.size(0))
            adj_fake_norm = torch.from_numpy(preprocess_graph(adj_fake.numpy()))

            z_mean, z_log_std = gcn_vae(adj_vae_norm, feat_all)
            z_mean_new, z_log_std_new = gcn_vae(adj_fake_norm, feat_all)

            z_mean[feat.size(0):, :] = z_mean_new[feat.size(0):, :]
            z_log_std[:feat.size(0), :] = z_log_std[:feat.size(0), :]

            adj_h = sample_reconstruction(z_mean, z_log_std)
            auc_gcn_fake = get_roc_auc_score(adj_all, adj_h, mask)
            ap_gcn_fake = get_average_precision_score(adj_all, adj_h, mask)


            print('AUC:')
            info = 'Original GCN auc: {:.6f}'.format(auc_gcn)
            print(info)
            logging.info(info)
            info = 'MLP auc: {:.6f}'.format(auc_mlp)
            print(info)
            logging.info(info)
            info = 'FAKE auc: {:.6f}'.format(auc_gcn_fake)
            print(info)


            print('AP:')
            info = 'Original GCN AP: {:.6f}'.format(ap_gcn)
            print(info)
            logging.info(info)
            info = 'MLP AP: {:.6f}'.format(ap_mlp)
            print(info)
            logging.info(info)
            info = 'FAKE AP: {:.6f}'.format(ap_gcn_fake)
            print(info)
            logging.info(info)




# ###### get embeddings for traing and test objects #####
# with torch.no_grad():
#
#     adj_feed_norm = torch.from_numpy(preprocess_graph(g_adj))
#     if isinstance(label_all, np.ndarray):
#         label_all = torch.from_numpy(label_all)
#     feat = torch.from_numpy(X)
#     feat_all = torch.from_numpy(X_all)
#
#     # gcn_step.eval()
#     # z_mean_old, z_log_std_old, z_mean_new, z_log_std_new = gcn_step(torch.from_numpy(g_adj), feat, feat_all[feat.size()[0]:, :])
#     #
#     # z_mean = torch.cat((z_mean_old, z_mean_new))
#     # z_log_std = torch.cat((z_log_std_old, z_log_std_new))
#
#     gcn_vae.eval()
#     adj_vae_norm = torch.eye(feat_all.size()[0])
#     adj_vae_norm[:feat.size()[0], :feat.size()[0]] = adj_feed_norm
#     z_mean, z_log_std = gcn_vae(adj_vae_norm, feat_all)
#
#     normal = torch.distributions.Normal(0, 1)
#     z = normal.sample(z_mean.size())
#     z = z * torch.exp(z_log_std) + z_mean
#     z = z.numpy()
#
#     new_z = z.copy()
#     def invert(p):
#         p = list(p)
#         return np.array([p.index(l) for l in range(len(p))])
#
#     new_x_idx = invert(x_idx)
#     new_z = z[new_x_idx, :]
#
#     test_idx = x_idx[z_mean_old.size(0)+1:]
#
#     ## TO DO
#     np.savetxt('../saved_models/ice_cream_gcn_embed.txt', new_z)
#     np.savetxt('../saved_models/test_idx_gcn_array.txt', test_idx)
