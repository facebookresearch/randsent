# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch.nn as nn
import torch
from torch.autograd import Variable

import numpy as np

from . import model_utils

class ESN(nn.Module):

    def __init__(self, params):
        super(ESN, self).__init__()

        self.n_inputs = params.input_dim
        self.n_reservoir = params.output_dim
        self.spectral_radius = params.spectral_radius
        self.leaky = params.leaky
        self.concat_inp = params.concat_inp
        self.stdv = params.stdv
        self.bidirectional = params.bidirectional
        self.sparsity = params.sparsity
        self.params = params

        if self.concat_inp:
            self.n_reservoir = self.n_reservoir - self.n_inputs

        #create parameters
        self.W = nn.Parameter(torch.Tensor(self.n_reservoir , self.n_reservoir))
        self.Win = nn.Parameter(torch.Tensor(self.n_reservoir , self.n_inputs))

        if self.bidirectional:
            self.W_rev = nn.Parameter(torch.Tensor(self.n_reservoir, self.n_reservoir))
            self.Win_rev = nn.Parameter(torch.Tensor(self.n_reservoir, self.n_inputs))

        #init parameters
        self.W.data.uniform_(-0.5, 0.5)
        if self.bidirectional:
            self.W_rev.data.uniform_(-0.5, 0.5)

        self.Win.data.uniform_(-self.stdv, self.stdv)
        if self.bidirectional:
            self.Win_rev.data.uniform_(-self.stdv, self.stdv)


        #echo state property
        if self.spectral_radius > 0:
            radius = np.max(np.abs(np.linalg.eigvals(self.W.data)))
            self.W.data = self.W.data * (self.spectral_radius / radius)

            if self.bidirectional:
                radius = np.max(np.abs(np.linalg.eigvals(self.W_rev.data)))
                self.W_rev.data = self.W_rev.data * (self.spectral_radius / radius)

        #make network
        self.input_layer = nn.Linear(self.n_inputs, self.n_reservoir, bias=False)
        self.input_layer.weight = self.Win

        self.recurrent_layer = nn.Linear(self.n_reservoir, self.n_reservoir, bias=False)
        self.recurrent_layer.weight = self.W

        self.position_enc = torch.nn.Embedding(1000, 300, padding_idx=0)
        self.position_enc.weight.data = model_utils.position_encoding_init(1000, 300)

        if self.bidirectional:
            self.input_layer_rev = nn.Linear(self.n_inputs, self.n_reservoir, bias=True)
            self.input_layer_rev.weight = self.Win_rev

            self.recurrent_layer_rev = nn.Linear(self.n_reservoir, self.n_reservoir, bias=True)
            self.recurrent_layer_rev.weight = self.W_rev

            self.position_enc_rev = torch.nn.Embedding(1000, 300, padding_idx=0)
            self.position_enc_rev.weight.data = model_utils.position_encoding_init(1000, 300)

        if self.sparsity > 0:
            self.sparse(self.recurrent_layer.weight.data, self.sparsity)

            if self.bidirectional:
                self.sparse(self.recurrent_layer_rev.weight.data, self.sparsity)

        if params.gpu:
            self.cuda()

    def sparse(self, tensor, sparsity):
        rows, cols = tensor.shape
        num_zeros = int(math.ceil(sparsity * rows))

        with torch.no_grad():
            for col_idx in range(cols):
                row_indices = torch.randperm(rows)
                zero_indices = row_indices[:num_zeros]
                tensor[zero_indices, col_idx] = 0

    def get_word_embs(self, input_seq, word_pos, batch, torev):
        out = self.params.lut(Variable(input_seq))
        if self.params.gpu:
            out = out.cuda()

        if self.params.gpu:
            word_pos = word_pos.cuda()
        if not torev:
            word_pos = self.position_enc(word_pos)
        else:
            word_pos = self.position_enc_rev(word_pos)

        lengths = [len(i) for i in batch]
        lengths = Variable(torch.from_numpy(np.array(lengths)))

        out = out.transpose(1,0)
        word_pos = word_pos.transpose(1,0)
        if self.params.pos_enc and self.params.pos_enc_concat:
            out = torch.cat([out, word_pos], dim=2)
        elif self.params.pos_enc:
            out += word_pos
        return out, lengths

    def esn(self, out, torev):
        hidden_states = torch.zeros(out.size()[1], out.size()[0], self.n_reservoir) #SxBxD
        curr_hid = Variable(torch.zeros(1, 1, self.n_reservoir))
        if self.params.gpu:
            curr_hid = curr_hid.cuda()
            hidden_states = hidden_states.cuda()

        curr_hid.expand(1, out.size()[0], self.n_reservoir).contiguous()
        for i in range(out.size()[1]):
            curr_embs = out[:,i,:]
            if not torev:
                hid_i = self.input_layer(curr_embs) + self.recurrent_layer(curr_hid)
            else:
                hid_i = self.input_layer_rev(curr_embs) + self.recurrent_layer_rev(curr_hid)
            if self.params.activation is not None:
                hid_i = self.params.activation(hid_i)
            hidden_states[i] = hid_i
            if i > 1 and self.leaky > 0:
                hidden_states[i] = (1 - self.leaky) * hidden_states[i] + (self.leaky) * hidden_states[i - 1]
            curr_hid = hidden_states[i]

        if self.concat_inp:
            out = torch.cat([hidden_states, out.transpose(1,0)], dim=2)
        else:
            out = hidden_states

        return out

    #returns matrix of BxSxD of hidden states
    def forward(self, batch, params):
        self.params.lut = params.lut
        self.params.word2id = params.word2id
        input_seq = torch.LongTensor(1000, len(batch)).zero_()
        input_seq_rev = torch.LongTensor(1000, len(batch)).zero_()
        word_pos = torch.LongTensor(1000, len(batch)).zero_()
        word_pos_rev = torch.LongTensor(1000, len(batch)).zero_()

        cur_max_seq_len = 0
        for i, l in enumerate(batch):
            j = 0
            for k, w in enumerate(l):
                input_seq[j][i] = self.params.word2id[w]
                word_pos[j][i] = (k+1)
                j += 1
            if j > cur_max_seq_len: cur_max_seq_len = j
        input_seq = input_seq[:cur_max_seq_len]
        word_pos = word_pos[:cur_max_seq_len]

        if self.bidirectional:
            #now do same for reverse
            for i, l in enumerate(batch):
                j = 0
                l.reverse()
                for k, w in enumerate(l):
                    input_seq_rev[j][i] = self.params.word2id[w]
                    word_pos_rev[j][i] = (k+1)
                    j += 1
                if j > cur_max_seq_len: cur_max_seq_len = j
            input_seq_rev = input_seq_rev[:cur_max_seq_len]
            word_pos_rev = word_pos_rev[:cur_max_seq_len]

            emb_fwd, lengths = self.get_word_embs(input_seq, word_pos, batch, False)
            emb_rev, lengths_rev = self.get_word_embs(input_seq_rev, word_pos_rev, batch, True)

            out_fwd = self.esn(emb_fwd, False)
            out_rev = self.esn(emb_rev, True)

            out = torch.cat([out_fwd, out_rev], dim=2)
        else:
            emb_fwd, lengths = self.get_word_embs(input_seq, word_pos, batch, False)
            out = self.esn(emb_fwd, False)

        if self.params.pooling == "mean":
            out = model_utils.mean_pool(out, lengths)
        elif self.params.pooling == "max":
            out = model_utils.max_pool(out, lengths)
        elif self.params.pooling == "min":
            out = model_utils.min_pool(out, lengths)
        elif self.params.pooling == "hier":
            out = model_utils.hier_pool(out, lengths)
        elif self.params.pooling == "sum":
            out = model_utils.sum_pool(out, lengths)
        else:
            print("Warning: no pooling operation specified!")

        return out

    def encode(self, batch, params):
        return self.forward(batch, params).cpu().detach().numpy()
