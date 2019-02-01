# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

import numpy as np

def position_encoding_init(n_position, emb_dim):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def sum_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_() # BxF
    for i in range(x.size(1)):
        out[i] = torch.sum(x[:lengths[i],i,:], 0)
    return out

def mean_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_() # BxF
    for i in range(x.size(1)):
        out[i] = torch.mean(x[:lengths[i],i,:], 0)
    return out

def max_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_() # BxF
    for i in range(x.size(1)):
        out[i,:] = torch.max(x[:lengths[i],i,:], 0)[0]
    return out

def min_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_() # BxF
    for i in range(x.size(1)):
        out[i] = torch.min(x[:lengths[i],i,:], 0)[0]
    return out

def hier_pool(x, lengths, n=5):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_() # BxF
    if x.size(0) <= n: return mean_pool(x, lengths) # BxF
    for i in range(x.size(1)):
        sliders = []
        if lengths[i] <= n:
            out[i] = torch.mean(x[:lengths[i],i,:], 0)
            continue
        for j in range(lengths[i]-n):
            win = torch.mean(x[j:j+n,i,:], 0, keepdim=True) # 1xN
            sliders.append(win)
        sliders = torch.cat(sliders, 0)
        out[i] = torch.max(sliders, 0)[0]
    return out

def pool(out, lengths, params):
    if params.pooling == "mean":
        out = mean_pool(out, lengths)
    elif params.pooling == "max":
        out = max_pool(out, lengths)
    elif params.pooling == "min":
        out = min_pool(out, lengths)
    elif params.pooling == "hier":
        out = hier_pool(out, lengths)
    elif params.pooling == "sum":
        out = sum_pool(out, lengths)
    else:
        raise ValueError("No valid pooling operation specified!")
    return out

def param_init(model, opts):
    if opts.init == "orthogonal":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p)
    elif opts.init == "sparse":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.sparse_(p, sparsity=0.1)
    elif opts.init == "normal":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.normal_(p)
    elif opts.init == "uniform":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, a=-0.1, b=0.1)
    elif opts.init == "kaiming":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
    elif opts.init == "xavier":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def embed(batch, params, se_params, position_enc=None, to_reverse=0):
    input_seq = torch.LongTensor(params.max_seq_len, len(batch)).zero_()
    word_pos = None

    if params.pos_enc:
        word_pos = torch.LongTensor(params.max_seq_len, len(batch)).zero_()

    cur_max_seq_len = 0
    for i, l in enumerate(batch):
        j = 0
        if to_reverse:
            l.reverse()
        for k, w in enumerate(l):
            if k == params.max_seq_len:
                break
            input_seq[j][i] = se_params.word2id[w]
            if params.pos_enc:
                word_pos[j][i] = (k + 1)
            j += 1
        if j > cur_max_seq_len:
            cur_max_seq_len = j

    input_seq = input_seq[:cur_max_seq_len]
    out = se_params.lut(input_seq)
    if params.gpu:
        out = out.cuda()

    if params.pos_enc:
        word_pos = word_pos[:cur_max_seq_len]
        if params.gpu:
            word_pos = word_pos.cuda()
        out += position_enc(word_pos)

    lengths = [len(i) if len(i) < params.max_seq_len else params.max_seq_len for i in batch]
    lengths = torch.from_numpy(np.array(lengths))

    return lengths, out, word_pos