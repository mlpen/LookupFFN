import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from .simHash import SimHash
from .lsh import LSH

class LSHSampledLayer(nn.Module):
    def __init__(self, hash_weight, layer_size, K, L, num_class, rehash_all):
        super(LSHSampledLayer, self).__init__()
        self.D = layer_size
        self.K = K
        self.L = L
        self.num_class = num_class
        self.hash_weight = hash_weight
        self.rehash_all = rehash_all

        self.params = nn.Linear(layer_size, num_class)
        self.params.bias = nn.Parameter(torch.Tensor(num_class, 1))
        self.init_weights(self.params.weight, self.params.bias)

        # Construct lsh using triplet weight
        self.lsh = None
        self.initializeLSH()

        self.last_removed = None

    def initializeLSH(self):
        self.lsh = LSH(SimHash(self.D+1, self.K, self.L, self.hash_weight), self.K, self.L)
        weight_tolsh = torch.cat((self.params.weight, self.params.bias), dim = 1)
        self.lsh.insert_multi(weight_tolsh, self.num_class)

    def init_weights(self, weight, bias):
        initrange = 0.05
        weight.data.uniform_(-initrange, initrange)
        bias.data.fill_(0)

    def insert_removed(self):
        if self.last_removed is not None:
            weight_tolsh = torch.cat((self.params.weight[self.last_removed, :], self.params.bias[self.last_removed]), dim = 1)

            for i, id in enumerate(self.last_removed):
                self.lsh.insert(id, weight_tolsh[i, None])

    def reset_lsh(self):
        self.lsh.clear()
        weight_tolsh = torch.cat((self.params.weight, self.params.bias), dim = 1)
        self.lsh.insert_multi(weight_tolsh, self.num_class)

    def rehash(self):
        if self.rehash_all:
            self.reset_lsh()
        else:
            self.insert_removed()
            
    def forward(self, x):

        N, _ = x.size()

        # Prepare query and sample weights
        query_tolsh = torch.cat((x, torch.ones(N).unsqueeze(dim = 1).to(x.device)), dim = 1)
        sid, hashcode = self.lsh.query_multi(query_tolsh.data, N)

        sample_ids = torch.tensor(list(sid)).to(x.device)

        # Compute logits with sampled weights
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse = False)
        sample_bias = self.params.bias.squeeze()[sample_ids]
        sample_product = x.matmul(sample_weights.t())
        sample_logits = sample_product + sample_bias

        if not self.rehash_all:
            for sample_id in list(sid):
                self.lsh.query_remove(torch.cat((self.params.weight[sample_id, :], self.params.bias[sample_id])).unsqueeze(0), sample_id)

            self.last_removed = list(sid)

        return sample_logits, sample_ids
