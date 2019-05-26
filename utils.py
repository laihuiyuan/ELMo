# -*- coding: UTF-8 -*-

import codecs
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def break_sent(sentence, max_seq_len):
    cur = 0
    ret = []
    sen_len = len(sentence)
    while cur < sen_len:
        if cur + max_seq_len + 5 >= sen_len:
            ret.append(sentence[cur: sen_len])
            break
        ret.append(sentence[cur: min(sen_len, cur + max_seq_len)])
        cur += max_seq_len
    return ret


def read_data(path, max_chars=None, max_seq_len=30):
    data = []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append('<bos>')
            for token in line.strip().split():
                if max_chars is not None and len(token) + 2 > max_chars:
                    token = token[:max_chars - 2]
                data.append(token)
            data.append('<eos>')
    dataset = break_sent(data, max_seq_len)
    return dataset

def one_mini_batch(x, word2id, char2id, max_char_token, oov='<oov>', pad='<pad>', sort=True):
    batch_size = len(x)
    lst = list(range(batch_size))
    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    lens = [len(x[i]) for i in lst]
    max_len = max(lens)

    if word2id is not None:
        oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
        assert oov_id is not None and pad_id is not None
        batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                batch_w[i][j] = word2id.get(x_ij, oov_id)
    else:
        batch_w = None

    if char2id is not None:
        bow_id, eow_id, oov_id, pad_id = char2id.get('<eow>', None), char2id.get('<bow>', None), \
                                         char2id.get(oov,None), char2id.get(pad, None)

        assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None

        max_chars = max_char_token
        assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars

        batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)

        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                batch_c[i][j][0] = bow_id
                if x_ij == '<bos>' or x_ij == '<eos>':
                    batch_c[i][j][1] = char2id.get(x_ij)
                    batch_c[i][j][2] = eow_id
                else:
                    for k, c in enumerate(x_ij):
                        batch_c[i][j][k + 1] = char2id.get(c, oov_id)
                    batch_c[i][j][len(x_ij) + 1] = eow_id
    else:
        batch_c = None

    masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

    for i, x_i in enumerate(x):
        for j in range(len(x_i)):
            masks[0][i][j] = 1
            if j + 1 < len(x_i):
                masks[1].append(i * max_len + j)
            if j > 0:
                masks[2].append(i * max_len + j)

    assert len(masks[1]) <= batch_size * max_len
    assert len(masks[2]) <= batch_size * max_len

    masks[1] = torch.LongTensor(masks[1])
    masks[2] = torch.LongTensor(masks[2])

    return batch_w, batch_c, lens, masks


def gen_mini_batches(x, word2id, char2id, max_char_token, batch_size=1000, shuffle=True, sort=True):
    lst = list(range(len(x)))
    if shuffle:
        random.shuffle(lst)

    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]

    sum_len = 0.0
    batches_w, batches_c, batches_lens, batches_masks = [], [], [], []
    size = batch_size
    nbatch = (len(x) - 1) // size + 1
    for i in range(nbatch):
        start_id, end_id = i * size, (i + 1) * size
        bw, bc, blens, bmasks = one_mini_batch(x[start_id: end_id], word2id, char2id, max_char_token,sort=sort)
        sum_len += sum(blens)
        batches_w.append(bw)
        batches_c.append(bc)
        batches_lens.append(blens)
        batches_masks.append(bmasks)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_w = [batches_w[i] for i in perm]
        batches_c = [batches_c[i] for i in perm]
        batches_lens = [batches_lens[i] for i in perm]
        batches_masks = [batches_masks[i] for i in perm]

    # logging.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / len(x)))
    return batches_w, batches_c, batches_lens, batches_masks
