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
from sklearn.metrics import roc_auc_score, average_precision_score
import copy
import pandas as pd
import networkx as nx

from UAT.utils_function import *
from UAT.early_stop import EarlyStopping
from UAT.AAGNN_model import *
from UAT.Autoencoder_model import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=int, default=1, help='CUDA training.')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--patient', type=int, default=10,
                    help='Number of epoch for patience on early_stop.')

parser.add_argument('--auto_epochs', type=int, default=200,
                    help='Number of epoch to train.')
parser.add_argument('--auto_lr', type=float, default=0.0002,
                    help='Initial learning rate.')  # 0.001
parser.add_argument('--auto_weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--auto_beta1', type=float, default=0.9,
                    help='')
parser.add_argument('--auto_beta2', type=float, default=0.999,
                    help='')
parser.add_argument('--auto_1st_layer', type=int, default=256,
                    help='')
parser.add_argument('--auto_2nd_layer', type=int, default=128,
                    help='')
parser.add_argument('--auto_dropout', type=float, default=0.1,
                    help='')

parser.add_argument('--AAGNN_ori_epochs', type=int, default= 100,
                    help='Number of epoch to train.')

parser.add_argument('--AAGNN_dec_epochs', type=int, default= 250,
                    help='Number of epoch to train.')
parser.add_argument('--AAGNN_lr', type=float, default=0.001,
                    help='Initial learning rate.')  # 0.001
parser.add_argument('--AAGNN_weight_decay', type=float, default= 5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--AAGNN_hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--AAGNN_dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')

# parser.add_argument('--dataset', default='BlogCatalog')
# parser.add_argument('--dataset', default='Flickr')
parser.add_argument('--dataset', default='pubmed')

# parser.add_argument('--model', default='model_sub_1_hop_emb')
# parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
# parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('cuda running...')

"""dataset preparation"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj_dense, features, gnds, G, degree_mat = load_data(args.dataset, args.AAGNN_hidden)
all_label = torch.from_numpy(gnds)
features = torch.FloatTensor(np.array(features))
degree_mat = torch.FloatTensor(np.array(degree_mat))
adj_dense = torch.FloatTensor(np.array(adj_dense))

folder = './test_emb/'
if not os.path.exists(folder):
    os.makedirs(folder)

auto_para_name = "auto_parameter" + str(args.dataset) + "_" + str(args.auto_epochs) + "_" + str(args.auto_2nd_layer) + ".pt"
AAGNN_ori_para_name = "AAGNN_ori_parameter" + str(args.dataset) + "_" + str(args.AAGNN_ori_epochs) + "_" + str(args.AAGNN_hidden) + ".pt"

""" Start of Autoencoder """

autoencoder_model = Autoencoder(input_size = features.shape[1],
                                first_layer_size = args.auto_1st_layer,
                                second_layer_size = args.auto_2nd_layer,
                                dropout_ratio = args.auto_dropout)

autoencoder_optimizer = optim.Adam(autoencoder_model.parameters(), lr=args.auto_lr,\
                                   betas=(args.auto_beta1, args.auto_beta2),eps=1e-8)
autoencoder_model.to(device)
features = features.to(device)

if not os.path.exists(auto_para_name):
    print("no auto parameter exists")

    early_stopping = EarlyStopping(patience=args.patient, verbose=True, path=auto_para_name)
    for epoch in range(args.auto_epochs):
        t = time.time()
        autoencoder_model.train()
        autoencoder_optimizer.zero_grad()
        # features_noisy = autoencoder_model.add_noise(features).to(device)
        _, output = autoencoder_model(features)
        emb_loss_train = nn.MSELoss()(output, features)
        emb_loss_train.backward()
        autoencoder_optimizer.step()

        # autoencoder_model.eval()
        # _, output_val = autoencoder_model(features)
        # loss_val = nn.MSELoss()(output_val, features)
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.30f}'.format(emb_loss_train.item()),
              'time: {:.4f}s'.format(time.time() - t))

    torch.save(autoencoder_model.state_dict(), auto_para_name)
    print("auto parameter saved")

else:
    print("auto parameter loaded")
    autoencoder_model.load_state_dict(torch.load(auto_para_name))

emb, features_dec = autoencoder_model(features)

emplifier = 0.5
de_emplifier = 1

final_features = (features * de_emplifier) + (features_dec * emplifier)

"""AAGNN"""
AAGNN_ori_model = AAGNN(feature_size=final_features.shape[1],
                  hidden_size=args.AAGNN_hidden,
                  dropout_ratio=args.AAGNN_dropout,
                  Graph_networkx=G)
AAGNN_ori_optimizer = optim.Adam(AAGNN_ori_model.parameters(), lr=args.AAGNN_lr, weight_decay=args.AAGNN_weight_decay)

AAGNN_ori_model.eval()
final_features = final_features.cpu()
polluted_train_emb = AAGNN_ori_model(final_features, adj_dense, degree_mat).detach().cpu().numpy()
polluted_train_embed = torch.FloatTensor(polluted_train_emb)
center_ori = torch.mean(polluted_train_embed, 0)

sps_outlier_list = []
for i in range(features.shape[0]):
    sps_outlier_list.append(anomaly_score(polluted_train_embed[i, :], center_ori).item())

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

AAGNN_ori_model.to(device)
emb = emb.to(device)
adj_dense = adj_dense.to(device)
degree_mat = degree_mat.to(device)
center_ori = center_ori.to(device)

final_features = final_features.to(device)

if not os.path.exists(AAGNN_ori_para_name):
    print("no AAGNN ori parameter exists")

    early_stopping = EarlyStopping(patience=args.patient, verbose=True, path=AAGNN_ori_para_name)

    t_total = time.time()
    for epoch in range(args.AAGNN_ori_epochs):
        t = time.time()
        AAGNN_ori_model.train()
        AAGNN_ori_optimizer.zero_grad()
        output = AAGNN_ori_model(final_features, adj_dense, degree_mat)
        GDN_loss_train = objecttive_loss_valid(output[train_final_indx], center_ori)
        # GDN_loss_train = objecttive_loss_valid(output, center)
        GDN_loss_train.backward(retain_graph=True)
        AAGNN_ori_optimizer.step()

        # evaluate in val set

        AAGNN_ori_model.eval()
        output_val = AAGNN_ori_model(final_features, adj_dense, degree_mat)
        loss_val = objecttive_loss_valid(output_val[valid_final_indx], center_ori)
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.30f}'.format(loss_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        early_stopping(loss_val, AAGNN_ori_model)  # validation loss
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    torch.save(AAGNN_ori_model.state_dict(), AAGNN_ori_para_name)
    print("AAGNN parameter saved")

else:
    print("AAGNN parameter loaded")
    AAGNN_ori_model.load_state_dict(torch.load(AAGNN_ori_para_name))

"""end of AAGNN for original features"""

# df = pd.DataFrame((features_dec.detach().cpu().numpy()).reshape(5196,-1))
# df.to_csv(folder + "/" + "features_dec.csv")
# print("csv done")

# df = pd.DataFrame((features.detach().cpu().numpy()).reshape(5196,-1))
# df.to_csv(folder + "/" + "features.csv")
# print("csv done")


# the obtained results contain all nodes
AAGNN_ori_model.eval()

final_features = final_features.cpu()
AAGNN_ori_model = AAGNN_ori_model.cpu()
adj_dense = adj_dense.cpu()
degree_mat = degree_mat.cpu()
emb = emb.cpu()
final_ori_emb = AAGNN_ori_model(final_features, adj_dense, degree_mat)
final_ori_emb = final_ori_emb.detach().cpu()
center_ori = center_ori.cpu()

""" AAGNN original score"""
center_ori_score = []
for i in range(final_ori_emb.shape[0]):
    center_ori_score.append(anomaly_score(final_ori_emb[i, :], center_ori).item())


anomaly_score = (np.array(center_ori_score))

roc_auc = roc_auc_score(gnds[total_test_indx], anomaly_score[total_test_indx])
roc_pr_area = average_precision_score(gnds[total_test_indx], anomaly_score[total_test_indx])

print('---auc: %.4f' % roc_auc)
print('---aupr: %.4f' % roc_pr_area)

