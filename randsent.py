# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os, sys

import torch
import torch.nn as nn

import numpy as np

import utils
from models import BOREP, ESN, RandLSTM

def prepare(params, samples):
    words = set([])
    for l in samples:
        for w in l:
            if w not in words:
                words.add(w)
    word2id = {w:i for i, w in enumerate(['<p>'] + list(words))}
    params.word2id = word2id
    params.lut = utils.load_vecs(params, word2id, zero=params.zero)
    if params.random_word_embeddings:
        utils.init_word_embeds(params.lut, params)
    return params

def batcher(params, batch):
    network = params['network']
    for n,i in enumerate(batch):
        if len(i) == 0:
            batch[n] = ['<p>']
    with torch.no_grad():
        vec = network.encode(batch, params)
    return vec

def get_results(params, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if params.gpu:
        torch.cuda.manual_seed(seed)

    network = None
    if params.model == "borep":
        network = BOREP(params)
    elif params.model == "lstm":
        network = RandLSTM(params)
    elif params.model == "esn":
        network = ESN(params)

    se = senteval.engine.SE({
        'task_path': os.path.join(params.senteval_path, 'data'),
        'word_emb_file': params.word_emb_file, 'word_emb_dim': params.word_emb_dim,
        'usepytorch': True, 'kfold': params.n_folds, 'feat_dim': senteval_feat_dim,
        'random_word_embeddings': params.random_word_embeddings, 'seed': seed,
        'batch_size': params.se_batch_size, 'network': network
    }, batcher, prepare)

    if params.task_type == "downstream":
        results = se.eval(['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKRelatedness',
                           'SICKEntailment', 'STSBenchmark'])
    else:
        results = se.eval(
            ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
             'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion'])
    return results

def consolidate(results, total_results):
    new_r = {}
    for task, result in results.items():
        if 'devacc' in result:
            dev, test = result['devacc'], result['acc']
            new_r[task] = (dev, test)
        elif 'devpearson' in result:
            dev, test = result['devpearson'], result['pearson']
            dev = dev if not np.isnan(dev) else 0.
            test = test if not np.isnan(test) else 0.
            new_r[task] = (dev*100, test*100)
    for task in new_r:
        if task not in total_results:
            total_results[task] = []
        total_results[task].append(new_r[task])
    return total_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RandSent - Random Sentence Representations")

    parser.add_argument("--model",
                help="Type of model to use (either borep, esn, or lstm, default borep).",
                        choices=["borep","esn", "lstm"], default="borep")
    parser.add_argument("--task_type",
                help="Type of task to try (either downstream or probing, default downstream).",
                        choices=["downstream", "probing"], default="downstream")
    parser.add_argument("--n_folds", type=int,
                help="Number of folds for cross-validation in SentEval (default 10).", default=10)
    parser.add_argument("--se_batch_size", type=int,
                help="Batch size for embedding sentences in SentEval (default 16).", default=8)
    parser.add_argument("--gpu", type=int, choices=[0,1],
                help="Whether to use GPU (default 0).", default=0)
    parser.add_argument("--senteval_path", type=str,
                help="Path to SentEval (default ./SentEval).", default="./SentEval")
    parser.add_argument("--word_emb_file", type=str,
                help="Path to word embeddings file (default ./glove.840B.300d.txt).", default="./glove.840B.300d.txt")
    parser.add_argument("--word_emb_dim", type=int,
                help="Dimension of word embeddings (default 300).", default=300)

    #Network parameters
    parser.add_argument("--input_dim", type=int, default=300,
                help="Output feature dimensionality (default 300).")
    parser.add_argument("--output_dim", type=int, default=4096,
                help="Output feature dimensionality (default 4096).")
    parser.add_argument("--max_seq_len", type=int, default=96,
                help="Sequence length (default 96).")
    parser.add_argument("--bidirectional", type=int, choices=[0,1], default=1,
                help="Whether to be bidirectional (default 1).")
    parser.add_argument("--init", type=str, choices=["none", "orthogonal", "sparse", "normal",
                                                     "uniform", "kaiming", "xavier"],
                help="Type of initialization to use (either none, orthogonal, sparse, normal, uniform, kaiming, "
                     "or xavier, default none).", default="none")
    parser.add_argument("--activation", type=str,
                        help="Activation function to apply to features (default none).", default=None)
    parser.add_argument("--pooling", choices=["min", "max", "mean", "hier", "sum"],
                help="Type of pooling (either min, max, mean, hier, or sum, default max).", default="max")

    #Embedding parameters
    parser.add_argument("--zero", type=int, choices=[0,1],
                help="Whether to initialize word embeddings to zero (default 1).", default=1)
    parser.add_argument("--pos_enc", type=int, choices=[0,1], default=0,
                help="Whether to do positional encoding (default 0).")
    parser.add_argument("--pos_enc_concat", type=int, choices=[0,1],
                        help="Whether to concat positional encoding to regular embedding (default 0).", default=0)
    parser.add_argument("--random_word_embeddings", type=int, choices=[0,1],
                help="Whether to load pretrained embeddings (default 0).", default=0)

    #Projection parameters
    parser.add_argument("--projection", type=str, choices=["none", "same"],
                help="Type of projection (either none or same, default same).", default="same")

    #ESN parameters
    parser.add_argument("--spectral_radius", type=float,
                help="Spectral radius for ESN (default 1.).", default=1.)
    parser.add_argument("--leaky", type=float,
                help="Fraction of previous state to leak for ESN (default 0).", default=0)
    parser.add_argument("--concat_inp", type=int, choices=[0,1],
                help="Whether to concatenate input to hidden state for ESN (default 0).", default=0)
    parser.add_argument("--stdv", type=float,
                help="Width of uniform interval to sample weights for ESN (default 1).", default=1.)
    parser.add_argument("--sparsity", type=float,
                help="Sparsity of recurrent weights for ESN (default 0).", default=0)

    #LSTM parameters
    parser.add_argument("--num_layers", type=int,
                        help="Number of layers for random LSTM (default 1).", default=1)

    print(" ".join(sys.argv))
    params, remaining_args = parser.parse_known_args()
    assert remaining_args == []

    senteval_feat_dim = params.output_dim if not params.bidirectional else 2*params.output_dim
    params.activation = eval(params.activation)() if \
            (params.activation is not None and eval(params.activation) is not None) \
            else None

    if params.pos_enc_concat:
        params.input_dim *= 2
    if params.concat_inp:
        senteval_feat_dim += params.input_dim

    sys.path.insert(0, params.senteval_path)
    import senteval

    seeds = [10, 100, 1000, 10000, 100000]
    total_results = {}
    for seed in seeds:
        results = get_results(params, seed)
        total_results = consolidate(results, total_results)
        torch.cuda.empty_cache()

    for task, result in total_results.items():
        dev = [i[0] for i in result]
        test = [i[1] for i in result]
        print("{0} | {1:0.2f} {2:0.2f} | {3:0.2f} {4:0.2f}".format(task, np.mean(dev), np.std(dev),
                                                              np.mean(test), np.std(test)))
