import math

import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(torch.ones(size=(input_size, output_size)))
        if bias:
            self.bias = Parameter(torch.ones(size=(1, output_size)).squeeze())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = torch.mm(x, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

class Decoder(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(torch.ones(size=(input_size, output_size)))
        if bias:
            self.bias = Parameter(torch.ones(size=(1, output_size)).squeeze())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = torch.mm(x, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


