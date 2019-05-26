# -*- coding: UTF-8 -*-

from __future__ import print_function
from __future__ import unicode_literals

import os
import codecs
import random

import torch
import torch.nn as nn

from source.elmo.utils import *
from source.elmo.models.layers import CNN
from source.elmo.models.layers import NllLoss, ElmoBilm

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


class Model(nn.Module):
    def __init__(self, word_embedder, char_embedder,n_class, encode,
                 char_dim, filters, n_highway, encode_layers,  activation, encode_dim,
                 cell_clip, proj_clip, projection_dim=150, dropout=0.5, use_cuda=False):
        super(Model, self).__init__()

        self.use_cuda = use_cuda
        self.dropout = dropout
        self.encode = encode

        self.token_embedder = CNN(word_embedder, char_embedder, char_dim, filters,
                                  n_highway, activation, projection_dim, use_cuda)

        # if config['encoder']['name'].lower() == 'lstm':
        #   self.encoder = lstmbi(config, use_cuda)
        if self.encode.lower() == 'elmo':
            self.encoder = ElmoBilm(encode_dim, encode_layers, cell_clip,
                                    proj_clip, projection_dim, self.dropout, use_cuda)

        self.output_dim = projection_dim
        self.nll_loss = NllLoss(self.output_dim, n_class)

    def forward(self, word_inp, chars_inp, mask_package):

        token_embedding = self.token_embedder(word_inp, chars_inp,
                                              (mask_package[0].size(0),
                                               mask_package[0].size(1)))

        token_embedding = F.dropout(token_embedding, self.dropout, self.training)

        if self.encode == 'elmo':
            mask = Variable(mask_package[0].cuda()).cuda() if self.use_cuda else Variable(mask_package[0])
            encoder_output = self.encoder(token_embedding, mask)
            encoder_output = encoder_output[1]

            # [batch_size, len, hidden_size]
        # elif encoder_name == 'lstm':
        #   encoder_output = self.encoder(token_embedding)
        else:
            raise ValueError('')

        encoder_output = F.dropout(encoder_output, self.dropout, self.training)
        forward, backward = encoder_output.split(self.output_dim, 2)

        word_inp = Variable(word_inp)
        if self.use_cuda:
            word_inp = word_inp.cuda()

        mask1 = Variable(mask_package[1].cuda()).cuda() if self.use_cuda else Variable(mask_package[1])
        mask2 = Variable(mask_package[2].cuda()).cuda() if self.use_cuda else Variable(mask_package[2])

        forward_x = forward.contiguous().view(-1, self.output_dim).index_select(0, mask1)

        forward_y = word_inp.contiguous().view(-1).index_select(0, mask2)

        backward_x = backward.contiguous().view(-1, self.output_dim).index_select(0, mask2)
        backward_y = word_inp.contiguous().view(-1).index_select(0, mask1)

        return self.nll_loss(forward_x, forward_y), self.nll_loss(backward_x, backward_y)

    def save_model(self, path, save_nll_loss):
        torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pkl'))
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pkl'))
        if save_nll_loss:
            torch.save(self.nll_loss.state_dict(), os.path.join(path, 'nll_loss.pkl'))

    def load_model(self, path):
        self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl')))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl')))
        self.nll_loss.load_state_dict(torch.load(os.path.join(path, 'nll_loss.pkl')))
