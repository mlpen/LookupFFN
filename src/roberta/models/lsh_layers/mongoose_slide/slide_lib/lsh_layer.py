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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

np.random.seed(1234)
torch.manual_seed(1234)


class LSHSampledLayer(nn.Module):
    def __init__(self, hash_weight, layer_size, K, L, num_class, rehash_all, proj_is_param):
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

        self.thresh_hash = SimHash(self.D+1, 1, self.L)
        self.thresh = 0.3
        self.hashcodes = self.thresh_hash.hash(torch.cat((self.params.weight, self.params.bias), dim = 1))

        self.alpha = 1.0
        self.last_removed = None

    def initializeLSH(self):
        self.lsh = LSH(SimHash(self.D+1, self.K, self.L, self.hash_weight), self.K, self.L)
        weight_tolsh = torch.cat((self.params.weight, self.params.bias), dim = 1)
        self.lsh.insert_multi(weight_tolsh.to(device).data, self.num_class)
    
    def setSimHash(self, seed, hashweight = None):
        print("update simhash")
        if(hashweight!=None):
            self.lsh.setSimHash(SimHash(self.D+1, self.K, self.L, hashweight))

    def rebuild(self):
        weight_tolsh = torch.cat((self.params.weight, self.params.bias), dim=1)
        check = self.thresh_hash.hash(weight_tolsh)
        distance = check - self.hashcodes
        if torch.sum(torch.abs(distance)) > self.thresh * distance.numel():
            self.lsh.clear()
            self.lsh.insert_multi(weight_tolsh.to(device).data, self.num_class)
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
        self.lsh.insert_multi(weight_tolsh.to(device).data, self.num_class)

    def forward(self, x):
        time1 = time.time()
        if self.training:
            if self.rehash_all:
                self.reset_lsh()
            else:
                self.insert_removed()
                self.rebuild()
        else:
            pass

        time2 = time.time()
        print('rehash', time2 - time1)

        N, _ = x.size()

        # Prepare query and sample weights
        query_tolsh = torch.cat((x, torch.ones(N).unsqueeze(dim = 1).to(device)), dim = 1)
        sid, hashcode = self.lsh.query_multi(query_tolsh.data, N)

        sample_ids = torch.tensor(list(sid)).to(x.device)

        # Compute logits with sampled weights
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True)
        sample_bias = self.params.bias.squeeze()[sample_ids]
        sample_product = x.matmul(sample_weights.t())
        sample_logits = sample_product + sample_bias

        time3 = time.time()
        print('sample weights', time3 - time2)

        """
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

        query = torch.cat((x, torch.ones(N).unsqueeze(dim = 1).to(device)), dim = 1)

        retrieved, _ = self.lsh.query_multi_mask(query, N, self.num_class)
        retrieved = retrieved.bool()

        positive_mask = ip_t1_mask & retrieved
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
        """

        # Start change

        t1 = 0.001
        t2 = 0.5
        t1_th = int(self.num_class * (1 - t1))  # Positive inner product rank
        t2_th = int(self.num_class * (1 - t2))  # Negative inner product rank
        t1_ip = torch.mean(torch.kthvalue(sample_product, t1_th)[0]).item()
        t1_ip = max(0.0, t1_ip)
        t2_ip = torch.mean(torch.kthvalue(sample_product, t2_th)[0]).item()

        positive_mask = sample_product > t1_ip
        negative_mask = sample_product < t2_ip

        row, column = torch.where(positive_mask == 1)
        p_x = query_tolsh[row].detach()
        p_weight = torch.cat((sample_weights[column], sample_bias[column]), dim = 1).detach()

        assert p_x.size()[0] == p_weight.size()[0]
        assert p_x.size()[1] == p_weight.size()[1]

        row, column = torch.where(negative_mask == 1)
        n_x = query_tolsh[row].detach()

        n_weight = torch.cat((sample_weights[column], sample_bias[column]), dim = 1).detach()

        assert n_x.size()[0] == n_weight.size()[0]
        assert n_x.size()[1] == n_weight.size()[1]

        # End change

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

        time4 = time.time()
        print('build + -', time4 - time3)

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
        triplet_loss = torch.mean(torch.max(triplet_loss, torch.zeros(triplet_loss.size()).to(x.device)))
    
        time5 = time.time()
        print('loss', time5 - time4)

        for sample_id in list(sid):
            self.lsh.query_remove(torch.cat((self.params.weight[sample_id, :], self.params.bias[sample_id])).unsqueeze(0), sample_id)

        self.last_removed = list(sid)
        
        time6 = time.time()
        print('remove', time6 - time5)

        return sample_logits, sample_ids, triplet_loss
