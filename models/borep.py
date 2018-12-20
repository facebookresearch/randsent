# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from . import model_utils

class RandProjection(nn.Module):

    def __init__(self, params):
        super(RandProjection, self).__init__()
        self.params = params
        self.max_seq_len = params.max_seq_len

        if self.params.projection == "unique":
            self.lprojs = []
            for _ in range(self.max_seq_len):
                self.lprojs.append(self.getProjection())
        elif self.params.projection == "same":
            self.lprojs = [self.getProjection()]

        if self.params.bidirectional and self.params.projection != "none":
            self.rprojs = []
            for _ in range(len(self.lprojs)):
                self.rprojs.append(self.getProjection())

        if params.pos_enc:
            self.position_enc = torch.nn.Embedding(self.max_seq_len+1, 300, padding_idx=0)
            self.position_enc.weight.data = model_utils.position_encoding_init(self.max_seq_len+1, 300)

        if params.gpu:
            self.cuda()

    def getProjection(self):
        proj = nn.Linear(self.params.input_dim, self.params.output_dim)
        if self.params.init == "orthogonal":
            nn.init.orthogonal_(proj.weight)
        elif self.params.init == "sparse":
            nn.init.sparse_(proj.weight, sparsity=0.1)
        elif self.params.init == "normal":
            nn.init.normal_(proj.weight, std=0.1)
        elif self.params.init == "uniform":
            nn.init.uniform_(proj.weight, a=-0.1, b=0.1)
        elif self.params.init == "kaiming":
            nn.init.kaiming_uniform_(proj.weight)
        elif self.params.init == "xavier":
            nn.init.xavier_uniform_(proj.weight)
        nn.init.constant_(proj.bias, 0)
        if self.params.gpu:
            return proj.cuda()
        else:
            return proj

    def uniqproj(self, x, lengths):
        """ Unique random projection """
        batch_sz, seq_len = x.size(1), x.size(0)
        out = Variable(torch.FloatTensor(seq_len, batch_sz, self.params.output_dim).zero_())
        for i in range(seq_len):
            left = self.lprojs[i](x[i])
            out[i] = left
        return out

    def sameproj(self, x, lengths):
        """ Identical random projection """
        batch_sz, seq_len = x.size(1), x.size(0)
        out = Variable(torch.FloatTensor(seq_len, batch_sz, self.params.output_dim).zero_())
        for i in range(seq_len):
            left = self.lprojs[0](x[i])
            out[i] = left
        return out

    def bi_uniqproj(self, x, lengths):
        """ Bidirectional unique random projection """
        batch_sz, seq_len = x.size(1), x.size(0)
        out = Variable(torch.FloatTensor(seq_len, batch_sz, 2 * self.params.output_dim).zero_())
        for i in range(seq_len):
            left = self.lprojs[i](x[i])
#            right = self.lprojs[i](x[seq_len-i-1])
            right = self.rprojs[i](x[seq_len-i-1])
            out[i] = torch.cat([left, right], 1)
        return out

    def bi_sameproj(self, x, lengths):
        """ Bidirectional identical random projection """
        batch_sz, seq_len = x.size(1), x.size(0)
        out = Variable(torch.FloatTensor(seq_len, batch_sz, 2 * self.params.output_dim).zero_())
        for i in range(seq_len):
            left = self.lprojs[0](x[i])
            right = self.rprojs[0](x[seq_len-i-1])
            out[i] = torch.cat([left, right], 1)
        return out

    def forward(self, batch, se_params):
        input_seq = torch.LongTensor(self.max_seq_len, len(batch)).zero_()
        if self.params.pos_enc:
            word_pos = torch.LongTensor(self.max_seq_len, len(batch)).zero_()
        cur_max_seq_len = 0

        for i, l in enumerate(batch):
            j = 0
            for k, w in enumerate(l[:self.max_seq_len]):
                input_seq[j][i] = se_params.word2id[w]
                if self.params.pos_enc:
                    word_pos[j][i] = (k+1)
                j += 1
            if j > cur_max_seq_len: cur_max_seq_len = j

        input_seq = input_seq[:cur_max_seq_len]
        out = se_params.lut(Variable(input_seq))
        if self.params.gpu:
            out = out.cuda()

        if self.params.pos_enc:
            word_pos = word_pos[:cur_max_seq_len]
            word_pos = word_pos
            if self.params.gpu:
                word_pos = word_pos.cuda()
            out += self.position_enc(word_pos)

        lengths = [len(i) for i in batch]
        lengths = Variable(torch.from_numpy(np.array(lengths)))

        # all functions below are SxBxEmbedSize -> SxBxFeatDim
        if self.params.bidirectional and self.params.projection == "unique":
            out = self.bi_uniqproj(out, lengths)
        elif self.params.bidirectional and self.params.projection == "same":
            out = self.bi_sameproj(out, lengths)
        elif not self.params.bidirectional and self.params.projection == "unique":
            out = self.uniqproj(out, lengths)
        elif not self.params.bidirectional and self.params.projection == "same":
            out = self.sameproj(out, lengths)

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
            out = self.params.activation(out)

        return out

    def encode(self, batch, params):
        return self.forward(batch, params).cpu().detach().numpy()

