import numpy as np
import torch
from sklearn import metrics

def link_split_mask(adj, mask_ratio=0.1, seed=999):
    adj_t = np.tril(adj)
    np.random.seed(seed)
    edge_lists = np.where(adj_t > 0)
    x_idx = np.random.permutation(len(edge_lists[0]))
    row_list_pos = edge_lists[0][x_idx][:int(0.1 * len(edge_lists[0]))]
    col_list_pos = edge_lists[1][x_idx][:int(0.1 * len(edge_lists[0]))]


    edge_lists = np.where(adj_t <= 0)
    neg_row = edge_lists[0][edge_lists[0] > edge_lists[1]]
    neg_col = edge_lists[1][edge_lists[0] > edge_lists[1]]
    x_idx = np.random.permutation(len(neg_row))
    row_list_neg = neg_row[x_idx][:len(row_list_pos)]
    col_list_neg = neg_col[x_idx][:len(row_list_pos)]

    row = np.concatenate((row_list_pos,row_list_neg))
    col = np.concatenate((col_list_pos, col_list_neg))

    return np.concatenate((row,col)), np.concatenate((col, row))

def sample_reconstruction(z_mean, z_log_std):
    num_nodes = z_mean.size()[0]
    normal = torch.distributions.Normal(0, 1)

    # sample z to approximiate the posterior of A
    z = normal.sample(z_mean.size())
    z = z * torch.exp(z_log_std) + z_mean
    adj_h = torch.mm(z, z.permute(1, 0))
    return adj_h

def get_roc_auc_score(adj, adj_h, mask):
    adj_n = adj[mask].numpy() > 0.9
    adj_h_n = adj_h[mask].sigmoid().numpy()
    return metrics.roc_auc_score(adj_n, adj_h_n)

def get_average_precision_score(adj, adj_h, mask):
    adj_n = adj[mask].numpy() > 0.9
    adj_h_n = adj_h[mask].sigmoid().numpy()
    return metrics.average_precision_score(adj_n, adj_h_n)

def get_equal_mask(adj_true, test_mask, thresh=0):
    """create a mask which gives equal number of positive and negtive edges"""
    adj_true = adj_true > thresh
    pos_link_mask = adj_true * test_mask
    num_links = int(pos_link_mask.sum().item())

    if num_links > 0.5 * test_mask.sum().item():
        raise ValueError('test nodes over connected!')

    neg_link_mask = (1 - adj_true) * test_mask
    neg_link_mask = neg_link_mask.numpy()
    row, col = np.where(neg_link_mask > 0)
    new_idx = np.random.permutation(len(row))
    row, col = row[new_idx][:num_links], col[new_idx][:num_links]
    neg_link_mask *= 0
    neg_link_mask[row, col] = 1
    neg_link_mask = torch.from_numpy(neg_link_mask)

    assert((pos_link_mask * neg_link_mask).sum().item() == 0)
    assert(neg_link_mask.sum().item() == num_links)
    assert(((pos_link_mask + neg_link_mask) * test_mask != (pos_link_mask + neg_link_mask)).sum().item() == 0)
    return pos_link_mask + neg_link_mask
