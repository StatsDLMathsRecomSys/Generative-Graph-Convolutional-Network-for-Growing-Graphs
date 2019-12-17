import numpy as np
import torch
import networkx as nx
import scipy.sparse as sp
from torch.utils.data import Dataset

def generate_random_graph_data(n_nodes=100, x_dim = 10, seed=999, noise=True, with_test=False, test_ratio=0.3, partial_ratio=None):
    np.random.seed(seed)
    X = np.random.rand(n_nodes * x_dim).reshape(n_nodes, x_dim) - 0.5
    X = X.astype(np.float32)

    gram = X.dot(X.T)
    if noise:
        gram += np.random.normal(size=gram.shape) * 0.01
    adj = gram > 0.5
    if partial_ratio is not None:
        X = X[:, :int(partial_ratio * X.shape[1])]

    # G = nx.from_numpy_matrix(adj)
    # largest_cc_node_list = list(max(nx.connected_components(G), key=len))
    # X_select = X[largest_cc_node_list, :]
    # adj_select = nx.to_numpy_matrix(G, nodelist=largest_cc_node_list)
    adj_select = adj
    X_select = X
    if with_test:
        cut_idx = int(adj_select.shape[0] * (1 - test_ratio))

        adj_train = adj_select[:cut_idx, :cut_idx]
        X_train = X_select[:cut_idx, :]
        return adj_train, X_train, adj_select, X_select
    else:
        return adj_select, X_select

def read_ice_cream(with_test=False, permute=False, test_ratio=0.3):
    features = sp.load_npz('../data/feat_sparse.npz')
    adj_orig = sp.load_npz('../data/P_sparse.npz')

    adj_orig = adj_orig + sp.eye(adj_orig.shape[0])
    adj_orig = adj_orig > 0.8

    X_select = features.todense().astype(np.float32)
    adj_select = adj_orig.todense().astype(np.float32)

    if permute:
        x_idx = np.random.permutation(adj_select.shape[0])
        adj_select = adj_select[np.ix_(x_idx, x_idx)]
        X_select = X_select[x_idx, :]

    if with_test:
        cut_idx = int(adj_select.shape[0] * (1 - test_ratio))

        adj_train = adj_select[:cut_idx, :cut_idx]
        X_train = X_select[:cut_idx, :]
        return adj_train, X_train, adj_select, X_select
    else:
        return adj_select, X_select

def bfs_seq(G, start_id):
    """
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    """
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output

def preprocess_graph_torch(adj):
    with torch.no_grad():
        rowsum = adj.sum(1)
        degree_mat_inv_sqrt = torch.diag(rowsum ** -.5)
        adj_normalized = adj.mm(degree_mat_inv_sqrt).t().mm(degree_mat_inv_sqrt)
        return adj_normalized



def preprocess_graph(adj):
    # adj = sp.coo_matrix(adj)
    # adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized.astype(np.float32)


class GraphSequenceBfsRandSampler(Dataset):
    def __init__(self, adj, X, num_permutation=10000, seed=None, fix=False):

        # self.adj = nx.to_numpy_matrix(G)
        self.adj = adj
        self.len = adj.shape[0]
        self.X = X
        self.num_permutation = num_permutation
        self.fix = fix

    def __len__(self):
        return self.num_permutation

    def __getitem__(self, idx):
        adj_copy = self.adj.copy()
        X_copy = self.X.copy()

        # initial permutation
        len_batch = adj_copy.shape[0]
        if not self.fix:
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            X_copy = X_copy[x_idx, :]


        adj_copy = adj_copy.astype(np.float32)
        X_copy = X_copy.astype(np.float32)

        return torch.from_numpy(adj_copy), torch.from_numpy(X_copy)
