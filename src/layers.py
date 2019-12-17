import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from graph_data import preprocess_graph, preprocess_graph_torch


class WeightTransitionLinearUnit(Module):
    def __init__(self, orig_dim, map_dim):
        super(WeightTransitionLinearUnit, self).__init__()
        self.linear_1 = torch.nn.Linear(map_dim, orig_dim)
        self.linear_2 = torch.nn.Linear(map_dim, map_dim)
        self.linear_3 = torch.nn.Linear(map_dim, map_dim)
        self.orig_dim = orig_dim
        self.map_dim = map_dim

        # self.bias = Parameter(torch.FloatTensor(map_dim))
        # self.weight = Parameter(torch.FloatTensor(orig_dim, map_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # torch.nn.init.xavier_normal_(self.weight)
        # self.bias.data.uniform_(-stdv, stdv)
        pass

    def forward(self, last_w, z_cov):
        z_cov = (z_cov - z_cov.mean())/ z_cov.std()
        hidden = F.relu(self.linear_1(z_cov).t())
        w_update = self.linear_2(hidden)
        update_gate = torch.sigmoid(self.linear_3(hidden))
        # w_update = torch.mm(self.weight, z_cov) + self.bias
        #w_update = (w_update - w_update.min()) / w_update.max() * 0.001

        # update_gate = torch.clamp(update_gate, min= 0, max= 1)
        #print(bound)
        w_update = torch.clamp(w_update, min= -0.1, max= 0.1)

        # print(last_w)
        # print(w_update)
        return (1 - update_gate) * last_w + w_update * update_gate



class RecursiveGraphConvolutionalNetwork(Module):


    def __init__(self, in_features, hidden_dim, out_features, bias=True, dropout=0.3):
        super(RecursiveGraphConvolutionalNetwork, self).__init__()

        self.dropout = dropout

        self.init_hidden_weight = Parameter(torch.FloatTensor(in_features, hidden_dim))
        self.init_hidden_bias = Parameter(torch.FloatTensor(hidden_dim))

        self.init_mean_weight = Parameter(torch.FloatTensor(hidden_dim, out_features))
        self.init_mean_bias = Parameter(torch.FloatTensor(out_features))

        self.init_log_std_weight = Parameter(torch.FloatTensor(hidden_dim, out_features))
        self.init_log_std_bias = Parameter(torch.FloatTensor(out_features))

        self.hidden_w_transition = WeightTransitionLinearUnit(in_features, hidden_dim)
        self.mean_w_transition = WeightTransitionLinearUnit(hidden_dim, out_features)
        self.log_std_w_transition = WeightTransitionLinearUnit(hidden_dim, out_features)

        self.reset_parameters()

    def init_all_weights(self):
        self.hidden_weight = self.init_hidden_weight + 0.0
        self.hidden_bias = self.init_hidden_bias + 0.0

        self.mean_weight = self.init_mean_weight + 0.0
        self.mean_bias = self.init_mean_bias + 0.0

        self.log_std_weight = self.init_log_std_weight + 0.0
        self.log_std_bias = self.init_log_std_bias + 0.0


    def convo_ops(self, input, adj):
        input = F.dropout(input, self.dropout, training=self.training)
        support = torch.mm(input, self.hidden_weight)
        hidden = F.relu(torch.spmm(adj, support) + self.hidden_bias)

        hidden = F.dropout(hidden, self.dropout, training=self.training)
        support_mean = torch.mm(hidden, self.mean_weight)
        mean = torch.spmm(adj, support_mean)
        support_std = torch.mm(hidden, self.log_std_weight)
        log_std = torch.spmm(adj, support_std)

        return mean, log_std

    def weight_transition(self, last_z, current_z):
        # compute the 'covariance' matrix for the difference of z
        z_diff = last_z - current_z
        z_cov = torch.mm(torch.t(z_diff), z_diff)
        # self.hidden_weight = self.hidden_w_transition(self.hidden_weight, z_cov) * 1.0
        self.mean_weight = self.mean_w_transition(self.mean_weight, z_cov) *  1.0
        self.log_std_weight = self.log_std_w_transition(self.log_std_weight, z_cov) * 1.0


    def forward(self, adj, input, update_size, input_new=None):
        # print(adj.size(0))
        # print(update_size)
        if adj.size(0) < update_size:
            raise ValueError('adj must be no less than update size!')

        self.init_all_weights()

        normal = torch.distributions.Normal(0, 1)

        # print(input.size())

        adj_h = torch.eye(update_size)
        last_z_mean, last_z_log_std = self.convo_ops(input[:update_size], adj_h)

        num_step = int(math.ceil(adj.size()[0] / update_size))
        # adj_frame = torch.eye(adj.size(0))

        z_prior, z_post = [], []
        z_prior.append((last_z_mean, last_z_log_std))
        for step in range(num_step - 1):
            start_idx = step * update_size
            end_idx = min(adj.size()[0], start_idx + update_size)
            adj_feed_norm = preprocess_graph_torch(adj[:end_idx, :end_idx])
            curr_z_mean, curr_z_log_std = self.convo_ops(input[:end_idx], adj_feed_norm)

            # cache the z for loss computation later
            z_post.append((curr_z_mean, curr_z_log_std))

            # sample z to approximiate the posterior
            current_z = curr_z_mean + normal.sample(curr_z_mean.size()) * torch.exp(curr_z_log_std)
            last_z = last_z_mean + normal.sample(last_z_mean.size()) * torch.exp(last_z_log_std)

            # update w_(t-1) to w_(t) based on difference between z_(t-1) and z_(t)

            self.weight_transition(last_z, current_z)

            # update the hypothetic z based on adj_h
            adj_ext = torch.eye(adj_feed_norm.size(0) + update_size)
            adj_ext[:end_idx, :end_idx] = adj_feed_norm
            next_end_idx = end_idx + update_size
            last_z_mean, last_z_log_std = self.convo_ops(input[:next_end_idx], adj_ext)
            z_prior.append((last_z_mean, last_z_log_std))

        if self.training:
            return z_prior, z_post
        else:
            adj_h_all = torch.eye(input.size(0) + input_new.size(0))
            adj_h_all[:adj.size(0), :adj.size(0)] = adj
            input_all = torch.cat((input, input_new))
            z_out_mean, z_out_log_std = self.convo_ops(input_all, adj_h_all)
            return z_out_mean, z_out_log_std




    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.init_hidden_weight.size(1))
        self.init_hidden_weight.data.uniform_(-stdv, stdv)
        self.init_hidden_bias.data.uniform_(-stdv, stdv)
        self.init_mean_weight.data.uniform_(-stdv, stdv)
        self.init_mean_bias.data.uniform_(-stdv, stdv)
        self.init_log_std_weight.data.uniform_(-stdv, stdv)
        self.init_log_std_bias.data.uniform_(-stdv, stdv)



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, act=lambda x: x, dropout=0.0):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, training = self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphVae(Module):
    def __init__(self, features_dim, hidden_dim, out_dim, bias=False, dropout=0.3):
        super(GraphVae, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.gc1 = GraphConvolution(features_dim, hidden_dim, bias=bias, dropout=dropout, act=F.relu)
        self.gc_mean = GraphConvolution(hidden_dim, out_dim, bias=bias, dropout=dropout)
        self.gc_log_std = GraphConvolution(hidden_dim, out_dim, bias=bias, dropout=dropout)

    def forward(self, adj, input):
        hidden = self.gc1(input, adj)
        z_mean = self.gc_mean(hidden, adj)
        z_log_std = self.gc_log_std(hidden, adj)
        return z_mean, z_log_std

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAE(Module):
    def __init__(self, features_dim, hidden_dim, out_dim, bias=False, dropout=0.3):
        super(GraphAE, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.gc1 = GraphConvolution(features_dim, hidden_dim, bias=bias, dropout=dropout, act=F.relu)
        self.gc_z = GraphConvolution(hidden_dim, out_dim, bias=bias, dropout=dropout)

    def forward(self, adj, input):
        hidden = self.gc1(input, adj)
        z = self.gc_z(hidden, adj)
        return z

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MLP(Module):
    def __init__(self, features_dim, hidden_dim, out_dim, bias=True, dropout=0.3):
        super(MLP, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.linear = torch.nn.Linear(features_dim, hidden_dim)
        self.z_mean = torch.nn.Linear(hidden_dim, out_dim)
        self.z_log_std = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, input):
        hidden = F.relu(self.linear(input))
        z_mean = F.dropout(self.z_mean(hidden), self.dropout, training=self.training)
        z_log_std = F.dropout(self.z_log_std(hidden), self.dropout, training=self.training)
        return z_mean, z_log_std

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class RecursiveGraphConvolutionStep(Module):
    """
    Given A, X and X_new, sample Z_new from P(Z_new|A_hat, X_new)
    """

    def __init__(self, features_dim, hidden_dim, out_dim, bias=True, dropout=0.3):
        super(RecursiveGraphConvolutionStep, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.gc1 = GraphConvolution(features_dim, hidden_dim, dropout=dropout, act=F.relu)
        self.gc_mean = GraphConvolution(hidden_dim, out_dim, dropout=dropout)
        self.gc_log_std = GraphConvolution(hidden_dim, out_dim, dropout=dropout)

    def forward(self, adj, input, input_new=None):
        hidden_old = self.gc1(input, adj)
        z_mean_old = self.gc_mean(hidden_old, adj)
        z_log_std_old = self.gc_log_std(hidden_old, adj)

        if input_new is not None:
            adj_new = torch.eye(input_new.size()[0])
            hidden_new = self.gc1(input_new, adj_new)
            z_mean_new = self.gc_mean(hidden_new, adj_new)
            z_log_std_new = self.gc_log_std(hidden_new, adj_new)
            return z_mean_old, z_log_std_old, z_mean_new, z_log_std_new
        else:
            return z_mean_old, z_log_std_old

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class RecursiveGraphConvolutionStepAddOn(Module):
    """
    Given A, X and X_new, sample Z_new from P(Z_new|A_hat, X_new)
    """

    def __init__(self, features_dim, hidden_dim, out_dim, random_add=False, bias=False, dropout=0.3):
        super(RecursiveGraphConvolutionStepAddOn, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.random_add = random_add

        self.gc1 = GraphConvolution(features_dim, hidden_dim, bias=bias, dropout=dropout, act=F.relu)
        self.gc_mean = GraphConvolution(hidden_dim, out_dim, bias=bias, dropout=dropout)
        self.gc_log_std = GraphConvolution(hidden_dim, out_dim, bias=bias, dropout=dropout)

    def forward(self, adj, input, input_new=None):
        if input_new is None:
            adj = adj.numpy()
            adj_norm = torch.from_numpy(preprocess_graph(adj))
            hidden = self.gc1(input, adj_norm)
            z_mean =self.gc_mean(hidden, adj_norm)
            z_log_std = self.gc_log_std(hidden, adj_norm)
            return z_mean, z_log_std

        num_total_nodes = input_new.size()[0] + input.size()[0]
        if self.training:
            with torch.no_grad():
                adj_new = torch.zeros(num_total_nodes, num_total_nodes)
                if self.random_add:
                    num_edges = float(((adj > 0).sum() - adj.shape[0]) / 2)
                    p0 = num_edges / (num_total_nodes ** 2) 
                    adj_new.bernoulli_(p0)
                    adj_new = adj_new - adj_new.tril()
                    adj_new = ((adj_new + adj_new.t()) > 0).float()
                    
                adj_new += torch.eye(num_total_nodes)
                adj_new[:input.size()[0], :input.size()[0]] = adj
                adj_new = adj_new.numpy()
                adj_norm = torch.from_numpy(preprocess_graph(adj_new))
                input_all = torch.cat((input, input_new))

            hidden = self.gc1(input_all, adj_norm)
            z_mean = self.gc_mean(hidden, adj_norm)
            z_log_std = self.gc_log_std(hidden, adj_norm)

            z_mean_old = z_mean[:input.size()[0], :]
            z_log_std_old = z_log_std[:input.size()[0], :]

            z_mean_new = z_mean[input.size()[0]:, :]
            z_log_std_new = z_log_std[input.size()[0]:, :]

            return z_mean_old, z_log_std_old, z_mean_new, z_log_std_new

        else:
            adj = adj.numpy()
            adj_norm = torch.from_numpy(preprocess_graph(adj))
            hidden_old = self.gc1(input, adj_norm)
            z_mean_old = self.gc_mean(hidden_old, adj_norm)
            z_log_std_old = self.gc_log_std(hidden_old, adj_norm)

            adj_new = torch.eye(input_new.size()[0])
            hidden_new = self.gc1(input_new, adj_new)
            z_mean_new = self.gc_mean(hidden_new, adj_new)
            z_log_std_new = self.gc_log_std(hidden_new, adj_new)
            return z_mean_old, z_log_std_old, z_mean_new, z_log_std_new

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphFuse(Module):
    def __init__(self, features_dim, hidden_dim, out_dim, bias=True, dropout=0.3):
        super(GraphFuse, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.dropout = dropout

        self.mixture_weight = Parameter(torch.FloatTensor(1))

        self.hidden_weight = Parameter(torch.FloatTensor(features_dim, hidden_dim))
        self.hidden_bias = Parameter(torch.FloatTensor(hidden_dim))

        self.mean_weight = Parameter(torch.FloatTensor(hidden_dim, out_dim))
        self.mean_bias = Parameter(torch.FloatTensor(out_dim))

        self.log_std_weight = Parameter(torch.FloatTensor(hidden_dim, out_dim))
        self.log_std_bias = Parameter(torch.FloatTensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_weight.size(1))
        self.mixture_weight.data.uniform_(-stdv, stdv)
        self.hidden_weight.data.uniform_(-stdv, stdv)
        self.hidden_bias.data.uniform_(-stdv, stdv)
        self.mean_weight.data.uniform_(-stdv, stdv)
        self.mean_bias.data.uniform_(-stdv, stdv)
        self.log_std_weight.data.uniform_(-stdv, stdv)
        self.log_std_bias.data.uniform_(-stdv, stdv)


    def convo_ops(self, input, adj):
        input = F.dropout(input, self.dropout, training=self.training)
        support = torch.mm(input, self.hidden_weight)
        hidden = F.relu(torch.spmm(adj, support) + self.hidden_bias)

        hidden = F.dropout(hidden, self.dropout, training=self.training)
        support_mean = torch.mm(hidden, self.mean_weight)
        mean = torch.spmm(adj, support_mean)
        support_std = torch.mm(hidden, self.log_std_weight)
        log_std = torch.spmm(adj, support_std)

        return mean, log_std

    def mlp_ops(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        hidden = torch.mm(input, self.hidden_weight)
        hidden = F.relu(hidden + self.hidden_bias)
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        mean = torch.mm(hidden, self.mean_weight) + self.mean_bias
        log_std = torch.mm(hidden, self.log_std_weight) + self.log_std_bias


        return mean, log_std

    def forward(self, input, adj=None):
        mixture_ratio = torch.sigmoid(self.mixture_weight)
        if adj is None:
            return self.mlp_ops(input)
        else:
            z_mean_gcn, z_log_std_gcn = self.convo_ops(input, adj)
            z_mean_mlp, z_log_std_mlp = self.mlp_ops(input)
            z_mean = z_mean_gcn * self.mixture_weight + z_mean_mlp * (1 - self.mixture_weight)
            z_log_std = z_log_std_gcn * mixture_ratio  + z_log_std_mlp * (1 - mixture_ratio)
            return z_mean, z_log_std

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphFuseSimple(Module):
    def __init__(self, n_nodes, features_dim, hidden_dim, out_dim, bias=True, dropout=0.3):
        super(GraphFuseV2, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.dropout = dropout

        self.mixture_weight = Parameter(torch.FloatTensor(1))

        self.hidden_weight = Parameter(torch.FloatTensor(features_dim, hidden_dim))
        self.hidden_bias = Parameter(torch.FloatTensor(hidden_dim))

        self.gcn_hidden_weight = Parameter(torch.FloatTensor(n_nodes, hidden_dim))
        self.gcn_hidden_bias = Parameter(torch.FloatTensor(hidden_dim))

        self.mean_weight = Parameter(torch.FloatTensor(hidden_dim, out_dim))
        self.mean_bias = Parameter(torch.FloatTensor(out_dim))

        self.log_std_weight = Parameter(torch.FloatTensor(hidden_dim, out_dim))
        self.log_std_bias = Parameter(torch.FloatTensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_weight.size(1))
        self.mixture_weight.data.uniform_(-stdv, stdv)
        self.gcn_hidden_weight.data.uniform_(-stdv, stdv)
        self.gcn_hidden_bias.data.uniform_(-stdv, stdv)
        self.hidden_weight.data.uniform_(-stdv, stdv)
        self.hidden_bias.data.uniform_(-stdv, stdv)
        self.mean_weight.data.uniform_(-stdv, stdv)
        self.mean_bias.data.uniform_(-stdv, stdv)
        self.log_std_weight.data.uniform_(-stdv, stdv)
        self.log_std_bias.data.uniform_(-stdv, stdv)


    def convo_ops(self, input, adj):
        input = F.dropout(input, self.dropout, training=self.training)
        support = torch.mm(input, self.gcn_hidden_weight)
        hidden = F.relu(torch.spmm(adj, support) + self.gcn_hidden_bias)

        hidden = F.dropout(hidden, self.dropout, training=self.training)
        support_mean = torch.mm(hidden, self.mean_weight)
        mean = torch.spmm(adj, support_mean)
        support_std = torch.mm(hidden, self.log_std_weight)
        log_std = torch.spmm(adj, support_std)

        return mean, log_std

    def mlp_ops(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        hidden = torch.mm(input, self.hidden_weight)
        hidden = F.relu(hidden + self.hidden_bias)
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        mean = torch.mm(hidden, self.mean_weight) + self.mean_bias
        log_std = torch.mm(hidden, self.log_std_weight) + self.log_std_bias

        return mean, log_std

    def forward(self, input, adj=None):
        mixture_ratio = torch.sigmoid(self.mixture_weight)
        if adj is None:
            return self.mlp_ops(input)
        else:
            gcn_input = torch.eye(adj.size(0))
            z_mean_gcn, z_log_std_gcn = self.convo_ops(gcn_input, adj)
            z_mean_mlp, z_log_std_mlp = self.mlp_ops(input)
            z_mean = z_mean_gcn * self.mixture_weight + z_mean_mlp * (1 - self.mixture_weight)
            z_log_std = z_log_std_gcn * mixture_ratio  + z_log_std_mlp * (1 - mixture_ratio)
            return z_mean, z_log_std

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
