import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from .simHash import SimHash
from .lsh import LSH

import time


class LSHSampledLayer(nn.Module):
    def __init__(self, hash_weight, layer_size, K, L, num_class, rehash_all):
        super(LSHSampledLayer, self).__init__()
        self.D = layer_size
        self.K = K
        self.L = L
        self.num_class = num_class
        self.hash_weight = hash_weight
        self.rehash_all = rehash_all

        self.store_query = True
        # last layer
        self.params = nn.Linear(layer_size, num_class)
        self.params.bias = nn.Parameter(torch.Tensor(num_class, 1))
        self.init_weights(self.params.weight, self.params.bias)

        # construct lsh using triplet weight
        self.lsh = None
        self.initializeLSH()

        if not self.rehash_all:
            self.last_removed = None
            self.thresh_hash = SimHash(self.D+1, 1, self.L)
            self.thresh = 0.3
            self.hashcodes = self.thresh_hash.hash(torch.cat((self.params.weight, self.params.bias), dim = 1))

        self.alpha = 1.0

    def initializeLSH(self):
        self.lsh = LSH(SimHash(self.D+1, self.K, self.L, self.hash_weight, proj_is_param = True), self.K, self.L)
        weight_tolsh = torch.cat((self.params.weight, self.params.bias), dim = 1)
        self.lsh.insert_multi(weight_tolsh, self.num_class)
    
    def setSimHash(self, seed, hashweight = None):
        if(hashweight != None):
            self.lsh.setSimHash(SimHash(self.D+1, self.K, self.L, hashweight))

    def rebuild(self):
        weight_tolsh = torch.cat((self.params.weight, self.params.bias), dim=1)
        check = self.thresh_hash.hash(weight_tolsh)
        distance = check - self.hashcodes
        if torch.sum(torch.abs(distance)) > self.thresh * distance.numel():
            self.lsh.clear()
            self.lsh.insert_multi(weight_tolsh, self.num_class)
            self.hashcodes = check
        else:
            pass

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
        sid, hashcode = self.lsh.query_multi(query_tolsh, N)

        sample_ids = torch.tensor(list(sid)).to(x.device)

        if sample_ids.numel() == 0:
            sample_ids = torch.matmul(torch.cat((self.params.weight, self.params.bias), dim = 1), query_tolsh.t()).sum(dim = -1).argmax().unsqueeze(0).to(x.device)
        
        # Compute logits with sampled weights
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse = False)
        sample_bias = self.params.bias.squeeze()[sample_ids]
        sample_product = x.matmul(sample_weights.t())
        sample_logits = sample_product + sample_bias

        # Mongoose
        product = torch.matmul(x, self.params.weight.t())

        t1 = 0.001
        t2 = 0.5
        t1_th = int(self.num_class * (1 - t1))  # Positive inner product rank
        t2_th = int(self.num_class * (1 - t2))  # Negative inner product rank
        t1_ip = torch.mean(torch.kthvalue(product, t1_th)[0]).item()
        t1_ip = max(0.0, t1_ip)
        t2_ip = torch.mean(torch.kthvalue(product, t2_th)[0]).item()

        # IP threshold mask
        ip_t1_mask = product > t1_ip
        ip_t2_mask = product < t2_ip

        query = torch.cat((x, torch.ones(N).unsqueeze(dim = 1).to(x.device)), dim = 1)

        retrieved, _ = self.lsh.query_multi_mask(query, N, self.num_class)
        retrieved = retrieved.bool()

        positive_mask = ip_t1_mask & (~retrieved)
        negative_mask = ip_t2_mask & retrieved

        num_negative = torch.sum(negative_mask).item()
        num_positive = torch.sum(positive_mask).item()

        row, column = torch.where(positive_mask == 1)
        p_x = query[row].detach()
        p_weight = torch.cat((self.params.weight[column], self.params.bias[column]), dim = 1).detach()

        assert p_x.size()[0] == p_weight.size()[0]
        assert p_x.size()[1] == p_weight.size()[1]

        row, column = torch.where(negative_mask == 1)
        n_x = query[row].detach()

        n_weight = torch.cat((self.params.weight[column], self.params.bias[column]), dim = 1).detach()

        assert n_x.size()[0] == n_weight.size()[0]
        assert n_x.size()[1] == n_weight.size()[1]

        # Down sample 
        if(num_positive < num_negative):
            random_perm = torch.randperm(num_negative)
            permute_id=random_perm[: int(num_positive)]
            n_x = n_x[permute_id]
            n_weight = n_weight[permute_id]
            num_negative = n_x.size()[0]
        else:
            random_perm = torch.randperm(num_positive)
            permute_id=random_perm[:int(num_negative)]
            p_x = p_x[permute_id]
            p_weight = p_weight[permute_id]
            num_positive = p_x.size()[0]

        p_x = p_x.detach()
        p_weight = p_weight.detach()
        n_x = n_x.detach()
        n_weight = n_weight.detach()

        p_x_proj = torch.matmul(p_x, self.lsh.func.rp)
        p_weight_proj = torch.matmul(p_weight, self.lsh.func.rp)
        n_x_proj = torch.matmul(n_x, self.lsh.func.rp)
        n_weight_proj = torch.matmul(n_weight, self.lsh.func.rp)

        sim_p = F.cosine_similarity(p_x_proj, p_weight_proj, dim = -1, eps = 1e-6)
        sim_n = F.cosine_similarity(n_x_proj, n_weight_proj, dim = -1, eps = 1e-6)

        triplet_loss = sim_n - sim_p + self.alpha
        triplet_loss = torch.nan_to_num(torch.mean(torch.max(triplet_loss, torch.zeros(triplet_loss.size()).to(x.device))))

        if not self.rehash_all:
            for sample_id in list(sid):
                self.lsh.query_remove(torch.cat((self.params.weight[sample_id, :], self.params.bias[sample_id])).unsqueeze(0), sample_id)

            self.last_removed = list(sid)

        return sample_logits, sample_ids, triplet_loss
