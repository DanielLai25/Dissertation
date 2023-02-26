from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os

import torch
import torch.optim as optim
import scipy
import scipy.io
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import copy
import pandas as pd
import networkx as nx

from UAT.utils_function import *
from UAT.AAGNN_model import *
from UAT.early_stop import EarlyStopping


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=int, default=1, help='CUDA training.')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epoch to train.')
parser.add_argument('--patient', type=int, default=10,
                    help='N umber of epoch for patience on early_stop.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')  # 0.001
parser.add_argument('--beta1', type=float, default=0.85,
                    help='')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2048,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')

# parser.add_argument('--dataset', default='BlogCatalog')
parser.add_argument('--dataset', default='Flickr')
# parser.add_argument('--dataset', default='pubmed')

parser.add_argument('--model', default='model_sub_1_hop_emb')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

name = str(args.dataset) + "_" + str(args.epochs) + "_" + str(args.hidden) + ".pt"
print(name)

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('cuda running...')

# Load data
dataset = args.dataset


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


def random_nodes_selection(return_node_num, total_node, seed):
    randomlist = [i for i in range(total_node)]
    random.seed(seed)
    random.shuffle(randomlist)
    return randomlist[:return_node_num], randomlist[return_node_num: total_node]


def train_test_splitting(total_node, seed, split_ratio):
    randomlist = [i for i in range(total_node)]
    random.seed(seed)
    random.shuffle(randomlist)

    break_point = int(total_node * split_ratio)
    return randomlist[:break_point], randomlist[break_point: total_node]


def anomaly_score(node_embedding, c):
    # anomaly score of an instance is calculated by
    # square Euclidean distance between the node embedding and the center c
    return torch.sum((node_embedding - c) ** 2)


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


adj_dense, features, gnds, G, degree_mat = load_data(args.dataset, args.hidden)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_label = torch.from_numpy(gnds)
features = torch.FloatTensor(np.array(features)).to(device)
degree_mat = torch.FloatTensor(np.array(degree_mat)).to(device)
adj_dense = torch.FloatTensor(np.array(adj_dense)).to(device)

# Model and optimizer
AAGNN_model = AAGNN(feature_size=features.shape[1],
                             hidden_size=args.hidden,
                             dropout_ratio=args.dropout,
                             Graph_networkx=G)

optimizer = optim.Adam(AAGNN_model.parameters(), lr=args.lr, \
                        betas = (args.beta1, args.beta2),
                        weight_decay=args.weight_decay)
AAGNN_model.to(device)

AAGNN_model.eval()
polluted_train_emb = AAGNN_model(features, adj_dense, degree_mat).detach().cpu().numpy()
polluted_train_embed = torch.FloatTensor(polluted_train_emb).to(device)
center = torch.mean(polluted_train_embed, 0).to(device)
# polluted_train_embed = torch.FloatTensor(polluted_train_emb)
# center = torch.mean(polluted_train_embed, 0)
##############################
# get high-likehood normal nodes
sps_outlier_list = []
for i in range(polluted_train_emb.shape[0]):
    sps_outlier_list.append(anomaly_score(polluted_train_embed[i, :], center).item())

sps_scores = np.array(sps_outlier_list)
sorted_sps_indices = np.argsort(-sps_scores, axis=0)  # high error node ranked top

# train_validation 50% & test split 50%
total_nodes_numb = features.shape[0]
# Data split: select top nodes
train_valid_num = int(0.5 * total_nodes_numb)  # 50% data used for train & valid
total_test_indx = sorted_sps_indices[0: -train_valid_num]  # top 50% of the data
train_valid_total_indx = sorted_sps_indices[-train_valid_num:]

# train: 30% total; valid: 20% total
train_vlid_number = train_valid_total_indx.shape[0]
train_seq, valid_seq = train_test_splitting(train_vlid_number, args.seed, split_ratio=(0.3/0.5))
train_final_indx = train_valid_total_indx[np.array(train_seq)]
valid_final_indx = train_valid_total_indx[np.array(valid_seq)]

gnd_1_indices = np.where(gnds[total_test_indx] == 1)[0]
test_outliers = total_test_indx[gnd_1_indices]
test_outliers_num = test_outliers.shape[0]
##############################
early_stopping = EarlyStopping(patience=args.patient, verbose=True, path=name)

if not os.path.exists(name):
    print("no parameter exists")

    t_total = time.time()
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        t = time.time()
        AAGNN_model.train()
        optimizer.zero_grad()
        output = AAGNN_model(features, adj_dense, degree_mat)
        loss_train = objecttive_loss_valid(output[train_final_indx], center)
        loss_train.backward()
        optimizer.step()

        # evaluate in val set
        AAGNN_model.eval()
        output_val = AAGNN_model(features, adj_dense, degree_mat)
        loss_val = objecttive_loss_valid(output_val[valid_final_indx], center)
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.6f}'.format(loss_train.item()),
              'time: {:.4f}s'.format(time.time() - t))

        early_stopping(loss_val, AAGNN_model)  # validation loss
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# load the last checkpoint with the best model
AAGNN_model.load_state_dict(torch.load(name))

# the obtained results contain all nodes
AAGNN_model.eval()
test_pred_emb = AAGNN_model(features, adj_dense, degree_mat)
y_true = np.array([label[0] for label in gnds])

tmp_list = []
for i in range(test_pred_emb.shape[0]):
    tmp_list.append(anomaly_score(test_pred_emb[i, :], center).item())


anomaly_score = np.array(tmp_list)
# only test on test data
roc_auc = roc_auc_score(gnds[total_test_indx], anomaly_score[total_test_indx])
roc_pr_area = average_precision_score(gnds[total_test_indx], anomaly_score[total_test_indx])

print('---auc: %.4f' % roc_auc)
print('---aupr: %.4f' % roc_pr_area)

# ###############################################
all_results = []
data_setting = args.dataset.split('/')[0]
data_name = args.dataset
result_folder = './result/' + data_setting
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

all_results.append(roc_auc)
all_results.append(roc_pr_area)
df = pd.DataFrame(np.array(all_results).reshape(-1, 2), columns=['AUC', 'AUPR'])
df.to_csv(result_folder + '/' + data_name + '_results_'+str(args.seed)+'.csv')

time.sleep(5)
