# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from . import model_utils

class BOREP(nn.Module):

    def __init__(self, params):
        super(BOREP, self).__init__()
        self.params = params
        self.max_seq_len = params.max_seq_len

        self.projection = params.projection
        self.proj = self.get_projection()

        self.position_enc = None
        if params.pos_enc:
            self.position_enc = torch.nn.Embedding(self.max_seq_len + 1, params.word_emb_dim, padding_idx=0)
            self.position_enc.weight.data = model_utils.position_encoding_init(self.max_seq_len + 1,
                                                                               params.word_emb_dim)

        if params.gpu:
            self.cuda()

    def get_projection(self):
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
            proj = proj.cuda()
        return proj

    def borep(self, x):
        batch_sz, seq_len = x.size(1), x.size(0)
        out = torch.FloatTensor(seq_len, batch_sz, self.params.output_dim).zero_()
        for i in range(seq_len):
            if self.projection:
                emb = self.proj(x[i])
            else:
                emb = x[i]
            out[i] = emb
        return out

    def forward(self, batch, se_params):
        lengths, out, word_pos = model_utils.embed(batch, self.params, se_params,
                                                   position_enc=self.position_enc, to_reverse=0)

        out = self.borep(out)
        out = model_utils.pool(out, lengths, self.params)

        if self.params.activation is not None:
            out = self.params.activation(out)

        return out

    def encode(self, batch, params):
        return self.forward(batch, params).cpu().detach().numpy()