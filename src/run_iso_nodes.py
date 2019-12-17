#!/Users/d0x00ar/anaconda3/bin/python3.6
import math
import argparse
import os
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from layers import RecursiveGraphConvolutionStep, RecursiveGraphConvolutionStepAddOn, GraphVae, MLP
from graph_data import generate_random_graph_data, GraphSequenceBfsRandSampler, preprocess_graph
from loss import recursive_loss, reconstruction_loss, vae_loss, recursive_loss_with_noise
from utils import link_split_mask, sample_reconstruction, get_roc_auc_score, get_average_precision_score, get_equal_mask

#os.chdir("/Users/d0x00ar/Documents/GitHub/R-GraphVAE/src")
import logging

#meta
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

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=400)
parser.add_argument('--out_dim', type=int, default=200)
parser.add_argument('--update_ratio', type=float, default=0.33)
parser.add_argument('--data_set', type=str, default='cora', choices = ['cora', 'citeseer', 'pubmed'])
parser.add_argument('--seed', default=None)
parser.add_argument('--refit', type=int, default=0)
parser.add_argument('--permute', type=int, default=1)
args = parser.parse_args()


hidden_dim = args.hidden_dim
out_dim = args.out_dim
cite_data = args.data_set

g_adj, X, g_adj_all, X_all = read_citation_dat(cite_data, with_test=True, permute=False)
num_nodes = g_adj_all.shape[0]
num_edges = ((g_adj_all > 0).sum() - num_nodes) / 2
print([num_nodes, num_edges])

size_update = int(num_nodes * args.update_ratio * 0.7)

seed = float(args.seed) if args.seed else None
unseen = True
refit = args.refit > 0
permute = args.permute > 0

norm=None
special = 'nodropout_DEBUG'

filename = '_'.join(['equal_size_cite', special, cite_data,
                    'size', str(size_update),
                    'hidden', str(hidden_dim),
                    'out', str(out_dim),
                    'fix', str(seed is not None),
                    'unseen', str(unseen),
                    'refit', str(refit),
                    'norm', str(norm),
                    'permute', str(permute),
                    'seed', str(seed)])
filename = '../data/exp_results/' + filename


logging.basicConfig(level=logging.DEBUG, filename=filename,
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

if seed is not None:
    np.random.seed(seed)

features_dim = X.shape[1]

dataset = GraphSequenceBfsRandSampler(g_adj, X, num_permutation=400, seed=seed, fix=False)

params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 2}

dataloader = torch.utils.data.DataLoader(dataset, **params)

# gcn_step = RecursiveGraphConvolutionStep(features_dim, hidden_dim, out_dim)
gcn_step = RecursiveGraphConvolutionStepAddOn(features_dim, hidden_dim, out_dim, dropout=0.0)
gcn_vae = GraphVae(features_dim, hidden_dim, out_dim, dropout=0)
mlp = MLP(features_dim, hidden_dim, out_dim, dropout=0)

optimizer = torch.optim.Adam(gcn_step.parameters(), lr=1e-3)
optimizer_vae = torch.optim.Adam(gcn_vae.parameters(), lr=1e-3)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
train_loss = 0

print('checkpoint1')

for batch_idx, (adj, feat) in enumerate(dataloader):
    adj = adj[0]
    feat = feat[0]

    if adj.size()[0] <= size_update:
        print("sample size {} too small, skipped!".format(adj.size()[0]))
        continue

    # train R-GCN

    optimizer.zero_grad()
    gcn_step.train()
    # loss = recursive_loss(gcn_step, adj, feat, size_update)
    loss = recursive_loss_with_noise(gcn_step, adj, feat, size_update, norm)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()

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
    mlp_train_loss = vae_loss(z_mean, z_log_std, adj, norm)
    mlp_train_loss.backward()
    optimizer_mlp.step()

    if batch_idx % 10 == 0:
        info ='R-GCN [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader),
                loss.item())
        print(info)
        logging.info(info)

        info = 'GCN [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader),
                vae_train_loss.item())
        print(info)
        logging.info(info)

        info = 'MLP [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader),
                mlp_train_loss.item())
        print(info)
        logging.info(info)

        with torch.no_grad():
            adj_feed_norm = torch.from_numpy(preprocess_graph(g_adj))
            adj_truth_all = torch.from_numpy(g_adj_all.astype(np.float32))

            feat = torch.from_numpy(X)
            feat_all = torch.from_numpy(X_all)

            mask = torch.ones_like(adj_truth_all).type(torch.ByteTensor)
            mask[:feat.size()[0], :feat.size()[0]] = 0

            diag_mask = torch.eye(mask.size(0)).type(torch.ByteTensor)
            mask = mask * (1 - diag_mask)

            mask = get_equal_mask(adj_truth_all, mask)


            # # test r-gcn
            gcn_step.eval()
            z_mean_old, z_log_std_old, z_mean_new, z_log_std_new = gcn_step(torch.from_numpy(g_adj), feat, feat_all[feat.size()[0]:, :])
            z_mean = torch.cat((z_mean_old, z_mean_new))
            z_log_std = torch.cat((z_log_std_old, z_log_std_new))

            adj_h = sample_reconstruction(z_mean, z_log_std)
            if refit:
                adj_hat = (adj_h > 0).type(torch.FloatTensor)
                adj_hat[:feat.size(0), :feat.size(0)] = torch.from_numpy(g_adj)
                z_mean, z_log = gcn_step(adj_hat, feat_all)
                adj_h = sample_reconstruction(z_mean, z_log_std)


            test_loss = reconstruction_loss(adj_truth_all, adj_h, mask, test=True)
            auc_rgcn = get_roc_auc_score(adj_truth_all, adj_h, mask)
            ap_rgcn = get_average_precision_score(adj_truth_all, adj_h, mask)

            info = 'R-GCN test loss: {:.6f}'.format(test_loss)
            print(info)
            logging.info(info)


            # test original gcn
            gcn_vae.eval()
            adj_vae_norm = torch.eye(feat_all.size()[0])
            adj_vae_norm[:feat.size()[0], :feat.size()[0]] = adj_feed_norm
            z_mean, z_log_std = gcn_vae(adj_vae_norm, feat_all)
            adj_h = sample_reconstruction(z_mean, z_log_std)
            test_loss = reconstruction_loss(adj_truth_all, adj_h, mask, test=True)
            auc_gcn = get_roc_auc_score(adj_truth_all, adj_h, mask)
            ap_gcn = get_average_precision_score(adj_truth_all, adj_h, mask)

            info = 'Original GCN test loss: {:.6f}'.format(test_loss)
            print(info)
            logging.info(info)


            # test on mlp
            mlp.eval()
            z_mean, z_log_std = mlp(feat_all)
            adj_h = sample_reconstruction(z_mean, z_log_std)
            test_loss = reconstruction_loss(adj_truth_all, adj_h, mask, test=True)
            auc_mlp = get_roc_auc_score(adj_truth_all, adj_h, mask)
            ap_mlp = get_average_precision_score(adj_truth_all, adj_h, mask)
            info = 'MLP test loss: {:.6f}'.format(test_loss)
            print(info)
            logging.info(info)

            print('AUC:')
            info = 'R-GCN auc: {:.6f}'.format(auc_rgcn)
            print(info)
            logging.info(info)
            info = 'Original GCN auc: {:.6f}'.format(auc_gcn)
            print(info)
            logging.info(info)
            info = 'MLP auc: {:.6f}'.format(auc_mlp)
            print(info)
            logging.info(info)



            info = 'R-GCN AP: {:.6f}'.format(ap_rgcn)
            print(info)
            logging.info(info)
            info = 'Original GCN AP: {:.6f}'.format(ap_gcn)
            print(info)
            logging.info(info)
            info = 'MLP AP: {:.6f}'.format(ap_mlp)
            print(info)
            logging.info(info)
