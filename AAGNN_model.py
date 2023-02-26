import torch.nn as nn
import torch.nn.functional as F
from UAT.AAGNN_layer import *


class AAGNN(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, Graph_networkx):
        super(AAGNN, self).__init__()

        self.gc1 = GraphConvolution(feature_size, hidden_size, Graph_networkx)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, adj_matrix, degree_norm):
        # Network Encoder
        embeddings = self.gc1(x, adj_matrix, degree_norm)
        embeddings = self.dropout(embeddings)
        embeddings = self.relu(embeddings)

        return embeddings

class AAGNN_multi_avg(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, Graph_networkx):
        super(AAGNN_multi_avg, self).__init__()

        self.gc1 = GraphConvolution_multi_avg(feature_size, hidden_size, Graph_networkx)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, adj_matrix, degree_norm, num_avg):
        # Network Encoder
        embeddings = self.gc1(x, adj_matrix, degree_norm, num_avg)
        embeddings = self.dropout(embeddings)
        embeddings = self.relu(embeddings)

        return embeddings