import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
import scipy
import networkx as nx

# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1)) # D
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


# def preprocess_adj(adj):
#     """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
#     adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
#     return adj_normalized


# def accuracy(output, labels):
#     preds = output.max(1)[1].type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)


# def f1(output, labels):
#     preds = output.max(1)[1].type_as(labels)
#     f1 = f1_score(labels, preds, average='weighted')
#     return f1


# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)


# def euclidean_dist(x, y):
#     # x: N x D query
#     # y: M x D prototype
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     assert d == y.size(1)
#
#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)
#
#     return torch.pow(x - y, 2).sum(2)  # N x M

def load_data(data_source, hidden_dim):
    data = scipy.io.loadmat("./data/{}.mat".format(data_source))
    gnds = data["gnd"]
    attributes_sprse = sp.csr_matrix(data["Attributes"])
    adj_csr_matrix = sp.csr_matrix(data["Network"])
    Graph = nx.from_scipy_sparse_matrix(adj_csr_matrix)

    attributes = attributes_sprse.todense()
    adj_maitrx = adj_csr_matrix.todense()

    hidden_dimension = hidden_dim
    degree = adj_maitrx.sum(axis=1).reshape(-1, 1)
    degree_divition = 1 / degree
    degree_matrix = np.tile(degree_divition, hidden_dimension)

    return adj_maitrx, attributes, gnds, Graph, degree_matrix


# def random_nodes_selection(return_node_num, total_node, seed):
#     randomlist = [i for i in range(total_node)]
#     random.seed(seed)
#     random.shuffle(randomlist)
#     return randomlist[:return_node_num], randomlist[return_node_num: total_node]


def train_test_splitting(total_node, seed, split_ratio):
    randomlist = [i for i in range(total_node)]
    random.seed(seed)
    random.shuffle(randomlist)

    break_point = int(total_node * split_ratio)
    return randomlist[:break_point], randomlist[break_point: total_node]

def nor_loss(node_embedding_list, c):
    # normal loss is calculated by mean squared Euclidian distance of
    # the normal node embeddings to hypersphere center c
    s = 0
    num_node = node_embedding_list.size()[0]
    for i in range(num_node):
        s = s + anomaly_score(node_embedding_list[i], c)
    return s/num_node


def objecttive_loss_valid(normal_node_emb, c):
    Nloss = nor_loss(normal_node_emb, c)
    return Nloss

def anomaly_score(node_embedding, c, atten=False):
    # anomaly score of an instance is calculated by
    # square Euclidean distance between the node embedding and the center c
    if atten == False:
        return torch.sum((node_embedding - c) ** 2)

    elif atten == True:
        emb = ((node_embedding - c) ** 2)
        emb_exp = torch.exp(emb)

        emb_exp_sum = torch.sum(emb_exp)

        emb_exp_div = emb_exp / emb_exp_sum
        emb_score = emb.dot(emb_exp_div)

        return emb_score

def locality_anomaly_score(node_embedding_list, adj_matrix, degree_norm, run_num):

    for i in range(run_num):
        if i == 0:
            CNN_emb = torch.mm(adj_matrix, node_embedding_list)
            CNN_emb = CNN_emb * degree_norm
        else:
            CNN_emb = torch.mm(adj_matrix, CNN_emb)
            CNN_emb = CNN_emb * degree_norm

    score_tensor = ((node_embedding_list - CNN_emb) ** 2).detach().numpy()
    score_tensor_exp = np.exp(score_tensor)
    score_tensor_exp_sum = np.sum(score_tensor_exp, axis=0)
    score_tensor_exp_sum = score_tensor_exp_sum.reshape(1,-1)
    score_tensor_exp_div = score_tensor_exp / (score_tensor_exp_sum)

    local_score_list = score_tensor * score_tensor_exp_div
    local_score_list = (np.sum(local_score_list,axis=1)).squeeze()

    return local_score_list

def attension_mechanism(node_embedding, node_embedding2):
    y_size = node_embedding.shape[0]
    x_size = node_embedding.shape[1]
    score_tensor = (node_embedding - node_embedding2) ** 2
    score_tensor = score_tensor.detach()
    score_tensor_exp = np.exp(score_tensor)
    score_tensor_exp_sum = list(score_tensor_exp.sum(axis=1))
    score_tensor_exp_sum = (np.array(score_tensor_exp_sum * y_size)).reshape(y_size,x_size)
    score_tensor_exp_sum = np.transpose(score_tensor_exp_sum)
    score_tensor_exp_div = score_tensor * score_tensor_exp / score_tensor_exp_sum
    local_score_list = np.sum(score_tensor_exp_div, axis=1)
    local_score_list = local_score_list.squeeze()
    return local_score_list


def min_max_normalization(feature):

    min_score = torch.FloatTensor(torch.min(feature, 0)[0])
    max_score = torch.FloatTensor(torch.max(feature, 0)[0])
    max_score += 1e-8

    diff_score = torch.sub(max_score, min_score)

    min_score = min_score.reshape([-1,1])
    min_score_full = min_score.expand(min_score.shape[0], feature.shape[0]).t()

    diff_score = diff_score.reshape([-1,1])
    diff_score_full = diff_score.expand(diff_score.shape[0], feature.shape[0]).t()

    feature_t = torch.sub(feature, min_score_full)
    feature_t = (feature_t / diff_score_full) + 1e-10

    return feature_t

def expotential_normalization(feature):
    fea_exp = torch.exp(feature)
    return fea_exp

def attention_score(score1, score2, score3):
    ten1 = torch.FloatTensor(score1).reshape(-1,1)
    ten2 = torch.FloatTensor(score2).reshape(-1,1)
    ten3 = torch.FloatTensor(score3).reshape(-1,1)
    total_ten = torch.cat((ten1, ten2, ten3),1)

    sort_total_ten, _ = torch.sort(total_ten, descending=True)

    # fea_exp = torch.exp(total_ten)
    # fea_exp_sum = torch.sum(fea_exp, 1)
    # fea_exp_sum = fea_exp_sum.reshape([-1,1])
    # fea_exp_sum_full = fea_exp_sum.expand(fea_exp_sum.shape[0], total_ten.shape[1])
    #
    # fea_atten = fea_exp / fea_exp_sum_full
    # final_score = total_ten * fea_atten
    #
    # final_score = torch.sum(final_score,1).t()

    ten_4 = torch.ones(size=(sort_total_ten.shape[0],1))
    ten_5 = ten_4 * 0.9
    ten_6 = ten_4 * 0.8
    ratio_ten = torch.cat((ten_4, ten_5, ten_6),1)

    final_score = torch.sum(sort_total_ten * ratio_ten, 1).t()

    return final_score


def min_max_for_score(list):
    min_score = min(list)
    max_score = max(list)
    diff_score = max_score - min_score
    tmp = []

    for i in range(len(list)):
        tmp_score = (list[i] - min_score) / diff_score
        tmp.append(tmp_score)

    return tmp

def add_noise(inputs):
        noisy = inputs + torch.randn_like(inputs)
        noisy = torch.clip(noisy, 0., 1.1)
        return noisy