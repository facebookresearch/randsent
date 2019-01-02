# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from . import model_utils

class RandLSTM(nn.Module):

    def __init__(self, params):
        super(RandLSTM, self).__init__()
        self.params = params

        self.bidirectional = params.bidirectional
        self.max_seq_len = params.max_seq_len

        self.e_hid_init = torch.zeros(1, 1, params.output_dim)
        self.e_cell_init = torch.zeros(1, 1, params.output_dim)

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

    def lstm(self, inputs, lengths):
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

    def forward(self, batch, se_params):
        lengths, out, _ = model_utils.embed(batch, self.params, se_params)
        out = out.transpose(1, 0)

        out = self.lstm(out, lengths)
        out = out.transpose(1,0)

        out = model_utils.pool(out, lengths, self.params)

        if self.params.activation is not None:
            out = self.params.activation(out)

        return out

    def encode(self, batch, params):
        return self.forward(batch, params).cpu().detach().numpy()
