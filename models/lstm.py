# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import numpy as np

from . import model_utils

class RandLSTM(nn.Module):

    def __init__(self, params):
        super(RandLSTM, self).__init__()
        self.params = params

        self.position_enc = torch.nn.Embedding(1000, 300, padding_idx=0)
        self.position_enc.weight.data = model_utils.position_encoding_init(1000, 300)
        self.bidirectional = params.bidirectional

        self.e_hid_init = Variable(torch.zeros(1, 1, params.output_dim))
        self.e_cell_init = Variable(torch.zeros(1, 1, params.output_dim))

        self.output_dim = params.output_dim
        self.num_layers = params.num_layers
        self.lm = nn.LSTM(params.input_dim, params.output_dim, num_layers=self.num_layers,
                          bidirectional=bool(params.bidirectional), batch_first=True)

        self.bidirectional += 1

        if params.init != "none":
            model_utils.param_init(self, params)

        if params.gpu:
            self.e_hid_init = self.e_hid_init.cuda()
            self.e_cell_init = self.e_cell_init.cuda()
            self.cuda()

    def encode_batch(self, inputs, lengths):
        bsz, max_len, _ = inputs.size()
        in_embs = inputs
        lens, indices = torch.sort(lengths, 0, True)

        e_hid_init = self.e_hid_init.expand(1*self.num_layers*self.bidirectional, bsz, self.output_dim).contiguous()
        e_cell_init = self.e_cell_init.expand(1*self.num_layers*self.bidirectional, bsz, self.output_dim).contiguous()
        all_hids, (enc_last_hid, _) = self.lm(pack(in_embs[indices],
                                                        lens.tolist(), batch_first=True), (e_hid_init, e_cell_init))
        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]

        return all_hids

    def forward(self, batch, params):
        self.params.lut = params.lut
        self.params.word2id = params.word2id
        input_seq = torch.LongTensor(1000, len(batch)).zero_()
        word_pos = torch.LongTensor(1000, len(batch)).zero_()

        cur_max_seq_len = 0
        for i, l in enumerate(batch):
            j = 0
            for k, w in enumerate(l):
                input_seq[j][i] = self.params.word2id[w]
                word_pos[j][i] = (k+1)
                j += 1
            if j > cur_max_seq_len: cur_max_seq_len = j
        input_seq = input_seq[:cur_max_seq_len]
        out = self.params.lut(Variable(input_seq))
        if self.params.gpu:
            out = out.cuda()

        word_pos = word_pos[:cur_max_seq_len]
        word_pos = word_pos
        if self.params.gpu:
            word_pos = word_pos.cuda()
        word_pos = self.position_enc(word_pos)

        lengths = [len(i) for i in batch]
        lengths = Variable(torch.from_numpy(np.array(lengths)))

        out = out.transpose(1,0)
        word_pos = word_pos.transpose(1,0)
        if self.params.pos_enc and self.params.pos_enc_concat:
            out = torch.cat([out, word_pos], dim=2)
        elif self.params.pos_enc:
            out += word_pos

        out = self.encode_batch(out, lengths)
        out = out.transpose(1,0)

        # all functions below are SxBxFeatDim -> BxOutDim
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

        if self.params.activation is not None:
            if eval(self.params.activation) is not None:
                out = eval(self.params.activation)()(out)

        return out

    def encode(self, batch, params):
        return self.forward(batch, params).cpu().detach().numpy()
