# -*- coding: UTF-8 -*-

from __future__ import print_function
from __future__ import unicode_literals

import os, sys, codecs, argparse
import logging, json, collections, random

import torch
import torch.nn as nn
from source.elmo.utils import *
from source.elmo.models.layers import *


def model_config():
    
    parser = argparse.ArgumentParser()

    cmd = parser.add_argument_group("Data")
    cmd.add_argument('--gpu', type=int, default=0)
    cmd.add_argument('--char_dim', type=int, default=50)
    cmd.add_argument('--word_dim', type=int, default=300)
    cmd.add_argument('--dropout', type=int, default=0.1)
    cmd.add_argument('--max_char_token', type=int, default=10)
    cmd.add_argument('--batch_size', type=int, default=1000)
    cmd.add_argument('--encode_dim', type=int, default=1024)
    cmd.add_argument('--projection_dim', type=int, default=150)
    cmd.add_argument('--cell_clip', type=int, default=3)
    cmd.add_argument('--proj_clip', type=int, default=3)
    cmd.add_argument('--n_highway', type=int, default=2)
    cmd.add_argument('--encode_layers', type=int, default=2)
    cmd.add_argument('--encode_highway', type=int, default=2)
    cmd.add_argument("--encode", type=str, default="elmo")
    cmd.add_argument("--activation", type=str, default="relu")
    cmd.add_argument("--output", type=str, default="./source/elmo/output")
    cmd.add_argument("--embed_file", type=str, default="./data/cn.vector.txt")
    cmd.add_argument("--config_file", type=str, default="config.json")
    cmd.add_argument("--filters", type=list, default=[[1, 32], [2, 64], [3, 128], [4, 128]])

    config = parser.parse_args()

    return config

def gen_data(sentens, max_chars=None):
    dataset = []
    for line in sentens:
        data = ['<bos>']
        text = []
        for token in line.split():
            text.append(token)
            if max_chars is not None and len(token) + 2 > max_chars:
                token = token[:max_chars - 2]
            data.append(token)
        data.append('<eos>')
        dataset.append(data)

    return dataset

class Model(nn.Module):
    def __init__(self, encode, word_embedder, char_embedder,
                 char_dim, filters, n_highway, encode_layers, activation, encode_dim,
                 cell_clip, proj_clip,projection_dim=150, dropout=0.5, use_cuda=False):
        super(Model, self).__init__()

        self.use_cuda = use_cuda
        self.encode = encode

        self.token_embedder = CNN(word_embedder, char_embedder, char_dim, filters,
                                  n_highway, activation, projection_dim, use_cuda)

        if self.encode.lower() == 'elmo':
            self.encoder = ElmoBilm(encode_dim, encode_layers, cell_clip,
                                    proj_clip, projection_dim, dropout,use_cuda)
        else:
            self.encoder = BiLSTM(encode_dim, encode_layers, projection_dim, use_cuda)

        self.output_dim = projection_dim

    def forward(self, word_inp, chars_inp, mask_package):
        token_embedding = self.token_embedder(word_inp, chars_inp,
                                              (mask_package[0].size(0),
                                               mask_package[0].size(1)))

        if self.encode == 'elmo':
            mask = Variable(mask_package[0]).cuda() if self.use_cuda else Variable(mask_package[0])
            encoder_output = self.encoder(token_embedding, mask)
        else:
            encoder_output = self.encoder(token_embedding)

        return encoder_output[1]

    def load_model(self, path):
        self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl')))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl')))
        # self.classify_layer.load_state_dict(torch.load(os.path.join(path, 'nll_loss.pkl')))

class elmo_embed(nn.Module):
    def __init__(self,config):
        super(elmo_embed, self).__init__()

        self.config = config

        if self.config.gpu >= 0:
            torch.cuda.set_device(self.config.gpu)
        self.use_cuda = self.config.gpu >= 0 and torch.cuda.is_available()

        # For the model trained with character-based word encoder.
        if self.config.char_dim > 0:
            self.char_lexicon = {}
            with codecs.open(os.path.join(self.config.output, 'char.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.char_lexicon[token] = int(i)
            char_embedder = CharEmbed(self.config.char_dim, self.char_lexicon, fix_emb=False)
        else:
            char_embedder = None

        # For the model trained with word form word encoder.
        if self.config.word_dim > 0:
            self.word_lexicon = {}
            with codecs.open(os.path.join(self.config.output, 'word.dic'), 'r', encoding='utf-8') as fpi:
                for idx, line in enumerate(fpi):
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.word_lexicon[token] = int(i)
            word_embedder = WordEmbed(self.config.embed_file, self.config.word_dim,
                                      self.word_lexicon, fix_emb=False)
        else:
            word_embedder = None

        # instantiate the model
        self.model = Model(encode=self.config.encode, word_embedder=word_embedder,
                           char_embedder=char_embedder, char_dim=self.config.char_dim,
                           filters=self.config.filters, n_highway=self.config.n_highway,
                           encode_layers=self.config.encode_layers, activation=self.config.activation,
                           encode_dim=self.config.encode_dim, cell_clip=self.config.cell_clip,
                           proj_clip=self.config.proj_clip,projection_dim=self.config.projection_dim,
                           dropout=0.2,use_cuda=self.use_cuda)

        if self.use_cuda:
            self.model.cuda()
        self.model.load_model(self.config.output)

    def forward(self, inputs):

        test, text = gen_data(inputs, self.config.max_char_token)


        test_w, test_c, test_lens, test_masks = gen_mini_batches(
            test, self.word_lexicon, self.char_lexicon, self.config.max_char_token, 
            batch_size=1000, shuffle=False,sort=False)

        #self.model.eval()
        # embedding=[]
        for w, c, lens, masks in zip(test_w, test_c, test_lens, test_masks):
            output = self.model.forward(w, c, masks)
            # embedding.append(output)
        # embedding=torch.cat(embedding,dim=-1)
        # print(embedding[:,1:-1].shape)

        return output[:,1:-1].data

        

if __name__ == "__main__":
    # config=model_config()
    # print(config.filters[0])
    sent = ["我 爱你 中国", "你是 哪儿 的 人 ？","每个 孩子 都是 祖国 的 花朵"]
    emlo = gen_emded()
    em=emlo(sent)
    # if len(sys.argv) > 1 and sys.argv[1] == 'test':
    #   test_main()
    # else:
    #   print('Usage: {0} [test] [options]'.format(sys.argv[0]), file=sys.stderr)
