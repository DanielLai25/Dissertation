import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from UAT.Autoencoder_layer import Encoder, Decoder

class Autoencoder(nn.Module):
    def __init__(self, input_size, first_layer_size, second_layer_size, dropout_ratio):
        super(Autoencoder, self).__init__()

        self.Enc1 = Encoder(input_size, first_layer_size)
        self.Enc2 = Encoder(first_layer_size, second_layer_size)
        self.Dec1 = Decoder(second_layer_size, first_layer_size)
        self.Dec2 = Decoder(first_layer_size, input_size)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        encoded = self.Enc1(x)
        encoded = self.dropout(encoded)
        encoded = self.relu(encoded)

        encoded = self.Enc2(encoded)
        encoded = self.dropout(encoded)
        emb = self.relu(encoded)

        decoded = self.Dec1(emb)
        decoded = self.dropout(decoded)
        decoded = self.relu(decoded)

        decoded = self.Dec2(decoded)
        decoded = self.dropout(decoded)
        out = self.relu(decoded)

        return emb, out