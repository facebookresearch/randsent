# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

def load_vecs(params, word2id, zero=True):
    embs = nn.Embedding(len(word2id), params.word_emb_dim, padding_idx=word2id['<p>'])
    if zero:
        nn.init.constant_(embs.weight, 0.)

    matches = 0
    n_words = len(word2id)
    embedding_vocab = []
    with open(params.word_emb_file) as f:
        for line in f:
            word = line.split(' ', 1)[0]
            embedding_vocab.append(word)
    embedding_vocab = set(embedding_vocab)

    word_map = {}
    for word in word2id:
        if word in embedding_vocab:
            word_map[word] = word2id[word]
        else:
            new_word = word.lower().capitalize()
            if new_word in embedding_vocab:
                word_map[new_word] = word2id[word]
            else:
                new_word = word.lower()
                if new_word in embedding_vocab:
                    word_map[new_word] = word2id[word]

    with open(params.word_emb_file) as f:
        for line in f:
            word = line.split(' ', 1)[0]
            if word != '<p>':
                if word in word_map:
                    glove_vect = torch.FloatTensor(list(map(float, line.split(' ', 1)[1].split(' '))))
                    embs.weight.data[word_map[word]][:300].copy_(torch.FloatTensor(glove_vect))

                    matches += 1
                    if matches == n_words: break
    return embs

def init_word_embeds(emb, opts):
    if opts.init == "none":
        emb.weight = nn.Embedding(emb.weight.size()[0], emb.weight.size()[1]).weight
    if opts.init == "orthogonal":
        nn.init.orthogonal_(emb.weight)
    elif opts.init == "sparse":
        nn.init.sparse_(emb.weight, sparsity=0.1)
    elif opts.init == "normal":
        nn.init.normal_(emb.weight, std=0.1)
    elif opts.init == "uniform":
        nn.init.uniform_(emb.weight, a=-0.1, b=0.1)
    elif opts.init == "kaiming":
        nn.init.kaiming_uniform_(emb.weight)
    elif opts.init == "xavier":
        nn.init.xavier_uniform_(emb.weight)
