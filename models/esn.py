# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch.nn as nn
import torch

import numpy as np

from . import model_utils

class ESN(nn.Module):

    def __init__(self, params):
        super(ESN, self).__init__()
        self.params = params

        self.bidirectional = params.bidirectional
        self.max_seq_len = params.max_seq_len

        self.n_inputs = params.input_dim
        self.n_reservoir = params.output_dim
        self.spectral_radius = params.spectral_radius
        self.leaky = params.leaky
        self.concat_inp = params.concat_inp
        self.stdv = params.stdv
        self.sparsity = params.sparsity

        if self.concat_inp:
            self.n_reservoir = self.n_reservoir - self.n_inputs

        self.W = nn.Parameter(torch.Tensor(self.n_reservoir , self.n_reservoir))
        self.Win = nn.Parameter(torch.Tensor(self.n_reservoir , self.n_inputs))
        self.W.data.uniform_(-0.5, 0.5)
        self.Win.data.uniform_(-self.stdv, self.stdv)

        if self.bidirectional:
            self.W_rev = nn.Parameter(torch.Tensor(self.n_reservoir, self.n_reservoir))
            self.Win_rev = nn.Parameter(torch.Tensor(self.n_reservoir, self.n_inputs))
            self.W_rev.data.uniform_(-0.5, 0.5)
            self.Win_rev.data.uniform_(-self.stdv, self.stdv)

        if self.spectral_radius > 0:
            radius = np.max(np.abs(np.linalg.eigvals(self.W.data)))
            self.W.data = self.W.data * (self.spectral_radius / radius)
            if self.bidirectional:
                radius = np.max(np.abs(np.linalg.eigvals(self.W_rev.data)))
                self.W_rev.data = self.W_rev.data * (self.spectral_radius / radius)

        self.input_layer = nn.Linear(self.n_inputs, self.n_reservoir, bias=False)
        self.input_layer.weight = self.Win
        self.recurrent_layer = nn.Linear(self.n_reservoir, self.n_reservoir, bias=False)
        self.recurrent_layer.weight = self.W

        if self.bidirectional:
            self.input_layer_rev = nn.Linear(self.n_inputs, self.n_reservoir, bias=True)
            self.input_layer_rev.weight = self.Win_rev
            self.recurrent_layer_rev = nn.Linear(self.n_reservoir, self.n_reservoir, bias=True)
            self.recurrent_layer_rev.weight = self.W_rev

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

    def esn(self, out, torev):
        hidden_states = torch.zeros(out.size()[1], out.size()[0], self.n_reservoir) #SxBxD
        curr_hid = torch.zeros(1, 1, self.n_reservoir)
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

    def forward(self, batch, se_params):
        lengths, emb_fwd, _ = model_utils.embed(batch, self.params, se_params)
        _, emb_rev, _ = model_utils.embed(batch, self.params, se_params, to_reverse=1)

        emb_fwd = emb_fwd.transpose(1, 0)
        emb_rev = emb_rev.transpose(1, 0)

        if self.bidirectional:
            out_fwd = self.esn(emb_fwd, False)
            out_rev = self.esn(emb_rev, True)
            out = torch.cat([out_fwd, out_rev], dim=2)
        else:
            out = self.esn(emb_fwd, False)

        out = model_utils.pool(out, lengths, self.params)
        return out

    def encode(self, batch, params):
        return self.forward(batch, params).cpu().detach().numpy()
