# RandSent

Various methods for random sentence representations, from the paper ''No Training Required: Exploring Random Encoders for Sentence Classification''.

## Getting started

If you don't have SentEval yet, download it: git clone https://github.com/facebookresearch/SentEval.git
Download the evaluation data and place SentEval in this directory.

To use `randsent.py`, you will need GloVe embeddings: http://nlp.stanford.edu/data/glove.840B.300d.zip
Place glove.840B.300d.txt in this directory, or provide the path using `--word_emb_path`.

## Requirements

RandSent requires Python 3, Pytorch 1.0 and Numpy.
## Examples

These are some example commands:

ESN:
`python -u randsent.py --model esn --pooling mean --pos_enc 0 --output_dim 2048 --zero 1 --spectral_radius 1.0 --leaky 0 --concat_inp 0 --stdv 0.1 --activation None --bidirectional 1 --sparsity 0.5 --gpu 1`

random projection:
`python -u randsent.py --model borep --pooling mean --projection same --pos_enc 0 --bidirectional 0 --output_dim 4096 --activation nn.ReLU --zero 1 --gpu 1`

random LSTM:
`python -u randsent.py --model lstm --pooling mean --pos_enc 0 --bidirectional 1 --output_dim 2048 --zero 1 --num_layers 1 --activation None --gpu 1`

## License
RandSent is CC-NC licensed, as found in the LICENSE file.
