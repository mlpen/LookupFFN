import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import os
import sys

from .gather_add_amp import autograd

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1, dtype = b.dtype, device = b.device)
    return torch.sum(mask * b, -1)

def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)

def mra_diffuse(inp, diffuse_order):
    if diffuse_order is None:
        return inp

    batch_size, num_table, table_size, vector_dim = inp.shape
    log_table_size = int(math.log2(table_size))
    reshape = [batch_size, num_table] + [2] * log_table_size + [vector_dim]
    inp = inp.reshape(*reshape)

    sum_coef = 1
    for index in range(1, log_table_size + 1):
        coef = (1 / ((index + 1) ** diffuse_order))
        sum_coef += coef

    components = [inp]
    for index in range(1, log_table_size + 1):
        inp = inp.mean(dim = - index - 1, keepdim = True)
        coef = (1 / ((index + 1) ** diffuse_order))
        components.append((coef / sum_coef) * inp)

    sum_components = 0
    for comp in reversed(components):
        sum_components = sum_components + comp

    out = sum_components.reshape(batch_size, num_table, table_size, vector_dim)
    return out

def compute_global_indexes(indexes, table_size):
    batch_size, sequence_length, num_table, query_size = indexes.shape
    batch_idx = torch.arange(batch_size, dtype = torch.int32, device = indexes.device)
    table_idx = torch.arange(num_table, dtype = torch.int32, device = indexes.device)
    global_indexes = (batch_idx[:, None, None, None] * num_table + table_idx[None, None, :, None]) * table_size + indexes
    # print(global_indexes.min(), global_indexes.max())
    return global_indexes

def weighted_lookup(indexes, weights, tables, compute_type):
    assert len(indexes.shape) == 4
    assert len(weights.shape) == 4
    assert len(tables.shape) == 4

    batch_size, sequence_length, num_table, query_size = indexes.shape
    _, num_table, table_size, vector_dim = tables.shape
    assert weights.shape[0] == batch_size
    assert weights.shape[1] == sequence_length
    assert weights.shape[2] == num_table
    assert weights.shape[3] == query_size
    assert tables.shape[0] == batch_size

    global_indexes = compute_global_indexes(indexes, table_size).reshape(batch_size * sequence_length, num_table * query_size)
    global_weights = weights.reshape(batch_size * sequence_length, num_table * query_size)
    global_table = tables.reshape(batch_size * num_table * table_size, vector_dim)

    global_outputs = autograd.weighted_gather_add(
        global_indexes, global_table, global_weights,
        compute_type = compute_type
    )
    outputs = global_outputs.reshape(batch_size, sequence_length, vector_dim)

    return outputs

def weighted_insert(indexes, weights, values, table_size, compute_type):
    assert len(indexes.shape) == 4
    assert len(weights.shape) == 4
    assert len(values.shape) == 3

    batch_size, sequence_length, num_table, query_size = indexes.shape
    _, _, vector_dim = values.shape
    assert weights.shape[0] == batch_size
    assert weights.shape[1] == sequence_length
    assert weights.shape[2] == num_table
    assert weights.shape[3] == query_size
    assert values.shape[0] == batch_size
    assert values.shape[1] == sequence_length

    global_table_size = batch_size * num_table * table_size
    global_indexes = compute_global_indexes(indexes, table_size).reshape(batch_size * sequence_length, num_table * query_size)
    global_weights = weights.reshape(batch_size * sequence_length, num_table * query_size)
    global_values = values.reshape(batch_size * sequence_length, vector_dim)

    global_table = autograd.weighted_scatter_add(
        global_indexes, global_values, global_weights, global_table_size,
        compute_type = compute_type
    )
    tables = global_table.reshape(batch_size, num_table, table_size, vector_dim)

    return tables


def compute_code_score(hash_scores, num_query_per_table):
    assert num_query_per_table == 1

    code_length = hash_scores.shape[-1]

    # code = [batch_size, sequence_length, num_table]
    code = 2 * (hash_scores > 0).to(hash_scores.dtype) - 1
    code = bin2dec((code + 1) / 2, code_length).int()

    # score = [batch_size, sequence_length, num_table]
    max_hash_scores = hash_scores.abs()
    denominator = torch.prod(1 + torch.exp(- 2 * max_hash_scores), dim = -1)
    score = max_hash_scores.sum(dim = -1) / denominator
    
    return code.unsqueeze(-1), score

# def compute_code_score(hash_scores, num_query_per_table):

#     batch_size, sequence_length, num_table, code_length = hash_scores.shape

#     if num_query_per_table > 1:
#         noise = torch.randint(
#             1, int(2 ** code_length),
#             size = (batch_size, sequence_length, num_table, num_query_per_table - 1),
#             dtype = torch.int, device = hash_scores.device)
#         sign_flips = dec2bin(noise, code_length)
#         sign_flips = 1 - 2 * sign_flips
#         # print("sign_flips", sign_flips.shape, sign_flips[0, 0, 0])

#         ones = torch.ones(batch_size, sequence_length, num_table, 1, code_length, dtype = hash_scores.dtype, device = hash_scores.device)
#         sign_flips = torch.cat([ones, sign_flips], dim = -2)
#     elif num_query_per_table == 1:
#         sign_flips = torch.ones(batch_size, sequence_length, num_table, 1, code_length, dtype = hash_scores.dtype, device = hash_scores.device)
#     else:
#         raise Exception()
#     # print("sign_flips", sign_flips.shape, sign_flips[0, 0, 0])

#     # positive_code = [batch_size, sequence_length, num_table, code_length]
#     positive_code = 2 * (hash_scores > 0).to(hash_scores.dtype) - 1

#     # max_hash_scores = [batch_size, sequence_length, num_table, code_length]
#     max_hash_scores = positive_code * hash_scores

#     # all_possible = torch.arange(int(2 ** code_length), dtype = torch.int, device = hash_scores.device)
#     # all_possible = dec2bin(all_possible, code_length)
#     # all_possible = 1 - 2 * all_possible
#     # print(all_possible)
#     #
#     # all_scores = (max_hash_scores[0, 0, 0, None, :] * all_possible).sum(dim = -1)
#     # print("test1", torch.exp(all_scores - all_scores.max()).sum(dim = -1))
#     # print("test2", torch.exp(all_scores - all_scores.max())[(2 ** torch.arange(code_length, dtype = torch.int, device = hash_scores.device)).long()])
#     # print("test3", F.softmax(all_scores)[(2 ** torch.arange(code_length, dtype = torch.int, device = hash_scores.device)).long()])

#     # positive_score = [batch_size, sequence_length, num_table]
#     positive_score = (max_hash_scores).sum(dim = -1)

#     # denominator = [batch_size, sequence_length, num_table]
#     denominator = torch.prod(1 + torch.exp(- 2 * max_hash_scores), dim = -1)

#     # print("hash_scores", hash_scores[0, 0, 0])
#     # print("max_hash_scores", max_hash_scores[0, 0, 0])
#     # print("positive_score", positive_score[0, 0, 0])
#     # print("denominator", denominator[0, 0, 0])

#     # code = [batch_size, sequence_length, num_table, num_query_per_table, code_length]
#     code = positive_code[:, :, :, None, :] * sign_flips

#     # score, numerator = [batch_size, sequence_length, num_table, num_query_per_table]
#     flipped_score = (code * hash_scores[:, :, :, None, :]).sum(dim = -1)
#     numerator = torch.exp(flipped_score - positive_score.unsqueeze(-1))

#     # print("flipped_score", flipped_score[0, 0, 0])
#     # print("numerator", numerator[0, 0, 0])

#     score = numerator / denominator.unsqueeze(-1) * flipped_score

#     # print("score", score[0, 0, 0])
#     # print("softmax", (numerator / denominator.unsqueeze(-1))[0, 0, 0])

#     code, score = code[:, :, :, :num_query_per_table], score[:, :, :, :num_query_per_table]

#     code = bin2dec((code + 1) / 2, code_length).int()
#     return code, score

def query_tables(hash_scores, tables, num_query_per_table, compute_type):
    batch_size, sequence_length, num_table, code_length = hash_scores.shape
    code, score = compute_code_score(hash_scores, num_query_per_table)
    return weighted_lookup(code, score, tables, compute_type)

def construct_tables(hash_scores, values, num_query_per_table, compute_type):
    batch_size, sequence_length, num_table, code_length = hash_scores.shape
    code, score = compute_code_score(hash_scores, num_query_per_table)
    return weighted_insert(code, score, values, int(2 ** code_length), compute_type)
