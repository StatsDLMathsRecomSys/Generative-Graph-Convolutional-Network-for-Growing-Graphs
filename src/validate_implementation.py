import math
import argparse
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from layers import GraphVae, MLP
from loss import reconstruction_loss, vae_loss


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=400)
parser.add_argument('--out_dim', type=int, default=200)
parser.add_argument('--num_iters', type=int, default=200)
parser.add_argument('--data_set', type=str, default='cora', choices = ['cora', 'citeseer', 'pubmed'])
parser.add_argument('--seed', type=int, default=888)
args = parser.parse_args()

hidden_dim = args.hidden_dim
out_dim = args.out_dim
cite_data = args.data_set

norm=None
num_iters = args.num_iters
seed = args.seed
np.random.seed(seed)

############## utility functions ##############
# def read_citation_dat(dataset):
#     '''
#     dataset: {'cora', 'citeseer', 'pubmed'}
#     '''
#
#     feat_fname = '../data/' + dataset + '_features.npz'
#     adj_fname = '../data/' + dataset + '_graph.npz'
#     features = sp.load_npz(feat_fname)
#     adj_orig = sp.load_npz(adj_fname)
#     adj_orig = adj_orig + sp.eye(adj_orig.shape[0])
#     return adj_orig, features
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("../data/test/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/test/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)

    #print(adj_normalized[:20, :].sum(1))

    return adj_normalized.astype(np.float32)

def get_roc_score(edges_pos, edges_neg, emb):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


############ prepare data ##############
adj, feat = load_data(args.data_set)

features_dim = feat.shape[1]

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

adj_label = adj_train + sp.eye(adj_train.shape[0])

adj_norm = torch.from_numpy(preprocess_graph(adj))
adj_label = torch.from_numpy(adj_label.todense().astype(np.float32))
feat = torch.from_numpy(feat.todense().astype(np.float32))

############## init model ##############
gcn_vae = GraphVae(features_dim, hidden_dim, out_dim, bias=False, dropout=0.0)
optimizer_vae = torch.optim.Adam(gcn_vae.parameters(), lr=0.01)

mlp = MLP(features_dim, hidden_dim, out_dim, dropout=0.0)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-2)

for batch_idx in range(num_iters):
    # train GCN
    optimizer_vae.zero_grad()
    gcn_vae.train()
    z_mean, z_log_std = gcn_vae(adj_norm, feat)
    vae_train_loss = vae_loss(z_mean, z_log_std, adj_label)
    vae_train_loss.backward()
    optimizer_vae.step()

    #train mlp
    optimizer_mlp.zero_grad()
    mlp.train()
    z_mean, z_log_std = mlp(feat)
    mlp_train_loss = vae_loss(z_mean, z_log_std, adj_label)
    mlp_train_loss.backward()
    optimizer_mlp.step()
    print('GCN [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx, num_iters,
            100. * batch_idx / num_iters,
            vae_train_loss.item()))

    if batch_idx % 10 == 0:
        # print('GCN [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx, num_iters,
        #         100. * batch_idx / num_iters,
        #         vae_train_loss.item()))
        # print('MLP [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx, num_iters,
        #         100. * batch_idx / num_iters,
        #         mlp_train_loss.item()))

        with torch.no_grad():

            # test original gcn
            gcn_vae.eval()
            z_mean, z_log_std = gcn_vae(adj_norm, feat)

            normal = torch.distributions.Normal(0, 1)
            z = normal.sample(z_mean.size())
            z = z * torch.exp(z_log_std) + z_mean

            roc, ap = get_roc_score(val_edges, val_edges_false, z.numpy())
            print('GCN val AP: {:.6f}'.format(ap))
            print('GCN val AUC: {:.6f}'.format(roc))


            mlp.eval()
            z_mean, z_log_std = mlp(feat)
            normal = torch.distributions.Normal(0, 1)
            z = normal.sample(z_mean.size())
            z = z * torch.exp(z_log_std) + z_mean
            roc, ap = get_roc_score(val_edges, val_edges_false, z.numpy())
            print('MLP val AP: {:.6f}'.format(ap))
            print('MLP val AUC: {:.6f}'.format(roc))


with torch.no_grad():

    mlp.eval()
    z_mean, z_log_std = mlp(feat)
    normal = torch.distributions.Normal(0, 1)
    z = normal.sample(z_mean.size())
    z = z * torch.exp(z_log_std) + z_mean
    roc, ap = get_roc_score(test_edges, test_edges_false, z.numpy())
    print('MLP val AP: {:.6f}'.format(ap))
    print('MLP val AUC: {:.6f}'.format(roc))

    gcn_vae.eval()
    z_mean, z_log_std = gcn_vae(adj_norm, feat)
    normal = torch.distributions.Normal(0, 1)
    z = normal.sample(z_mean.size())
    z = z * torch.exp(z_log_std) + z_mean
    roc, ap = get_roc_score(test_edges, test_edges_false, z.numpy())
    print('GCN test AP: {:.6f}'.format(ap))
    print('GCN test AUC: {:.6f}'.format(roc))
