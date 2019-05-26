# -*- coding: UTF-8 -*-

from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import json,errno
import codecs,logging
import random, argparse
import numpy as np
from collections import  Counter

import torch
import torch.nn as nn
import torch.optim as optim

from source.elmo.models.elmo import *
from source.elmo.models.layers import *

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def model_config():

    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--model_path", type=str, default="./output")
    data_arg.add_argument("--config_file", type=str, default="config.json")
    data_arg.add_argument("--train_file", type=str, default="./data/corpus.txt")
    data_arg.add_argument("--embed_file", type=str, default="./data/cn.vector.txt")
    # data_arg.add_argument('--valid_path', type=str, default=None)
    # data_arg.add_argument('--test_path', type=str, default=None)
    
    #training
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--seed", type=int, default=1024)
    train_arg.add_argument("--gpu", type=int,default=-1)
    train_arg.add_argument("--lr", type=float, default=0.001)
    train_arg.add_argument("--lr_decay", type=float, default=0.9)
    train_arg.add_argument("--batch_size", type=int, default=64)
    train_arg.add_argument("--max_epoch", type=int, default=2)
    train_arg.add_argument("--clip_grad", type=float, default=5)
    train_arg.add_argument("--max_seq_len", type=int, default=30)
    train_arg.add_argument('--max_char_token', type=int, default=10)
    train_arg.add_argument('--char_dim', type=int, default=50)
    train_arg.add_argument('--n_highway', type=int, default=2)
    train_arg.add_argument('--encode_layers', type=int, default=2)
    train_arg.add_argument("--activation", type=str, default="relu")
    train_arg.add_argument('--encode_dim', type=int, default=1024)
    train_arg.add_argument('--cell_clip', type=int, default=3)
    train_arg.add_argument('--proj_clip', type=int, default=3)
    train_arg.add_argument('--word_dim', type=int, default=300)
    train_arg.add_argument('--projection_dim', type=int, default=150)
    train_arg.add_argument('--dropout', type=int, default=0.1)


    train_arg.add_argument("--min_count", type=int, default=1)
    train_arg.add_argument("--eval_steps", type=int,default=1024)
    train_arg.add_argument("--valid_size", type=int, default=64)
    train_arg.add_argument("--max_vocab_size", type=int, default=150000)
    train_arg.add_argument("--encode", type=str, default="elmo")

    train_arg.add_argument("--save_nll_loss", default=False, action='store_true')
    train_arg.add_argument("--optimizer", default='adam', choices=['sgd', 'adam', 'adagrad'])
    train_arg.add_argument("--filters", type=list, default=[[1, 32], [2, 32], [3, 128], [4, 128]])


    config = parser.parse_args()

    return config


def get_truncated_vocab(dataset, min_count):
    word_count = Counter()
    for sentence in dataset:
        word_count.update(sentence)

    word_count = list(word_count.items())
    word_count.sort(key=lambda x: x[1], reverse=True)

    for i, (word, count) in enumerate(word_count):
        if count < min_count:
            break

    # logging.info('Truncated word count: {0}.'.format(sum([count for word, count in word_count[i:]])))
    logging.info('Original vocabulary size: {0}.'.format(len(word_count)))
    return word_count[:i]

# def eval_model(model, valid):
#   model.eval()
#   if model.config['classifier']['name'].lower() == 'cnn_softmax' or \
#       model.config['classifier']['name'].lower() == 'sampled_softmax':
#     model.nll_loss.update_embedding_matrix()
#     #print('emb mat size: ', model.nll_loss.embedding_matrix.size())
#   total_loss, total_tag = 0.0, 0
#   valid_w, valid_c, valid_lens, valid_masks = valid
#   for w, c, lens, masks in zip(valid_w, valid_c, valid_lens, valid_masks):
#     loss_forward, loss_backward = model.forward(w, c, masks)
#     total_loss += loss_forward.data[0]
#     n_tags = sum(lens)
#     total_tag += n_tags
#   model.train()
#   return np.exp(total_loss / total_tag)


def train_epoch(epoch, config, model, configimizer,train, best_train):

    model.train()
    start_time = time.time()

    cnt = 0
    total_loss, total_tag = 0.0, 0

    train_w, train_c, train_lens, train_masks = train

    lst = list(range(len(train_w)))
    random.shuffle(lst)

    train_w = [train_w[l] for l in lst]
    train_c = [train_c[l] for l in lst]
    train_lens = [train_lens[l] for l in lst]
    train_masks = [train_masks[l] for l in lst]

    for w, c, lens, masks in zip(train_w, train_c, train_lens, train_masks):
        cnt += 1
        model.zero_grad()
        loss_forward, loss_backward = model.forward(w, c, masks)

        loss = (loss_forward + loss_backward) / 2.0
        total_loss += loss_forward.item()
        n_tags = sum(lens)
        total_tag += n_tags
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        configimizer.step()
        if cnt * config.batch_size % 5000 == 0:

            logging.info("Epoch={} step={} lr={:.6f} train_ppl={:.6f} time={:.2f}s".format(
                epoch, cnt, configimizer.param_groups[0]['lr'],
                np.exp(total_loss / total_tag), time.time() - start_time
            ))
            start_time = time.time()

        if cnt % config.eval_steps == 0 or cnt % len(train_w) == 0:
            # if valid is None:
            train_ppl = np.exp(total_loss / total_tag)
            logging.info("Epoch={} step={} lr={:.6f} train_ppl={:.6f}".format(
                epoch, cnt, configimizer.param_groups[0]['lr'], train_ppl))
            if train_ppl < best_train:
                best_train = train_ppl
                # logging.info("New record achieved on training dataset!")
                model.save_model(config.model_path, config.save_nll_loss)
    return best_train


def train():

    config = model_config()

    with open(config.config_file, 'r') as fin:
        elmo_config = json.load(fin)

    # set seed.
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if config.gpu >= 0:
        torch.cuda.set_device(config.gpu)
        if config.seed > 0:
            torch.cuda.manual_seed(config.seed)

    use_cuda = config.gpu >= 0 and torch.cuda.is_available()


    train_data = read_data(config.train_file, config.max_char_token, config.max_seq_len)
    logging.info('training instance: {}, training tokens: {}.'.format(len(train_data),
                                                                      sum([len(s) - 1 for s in train_data])))
    valid_data = None
    test_data = None

    word_lexicon = {}

    # Maintain the vocabulary. vocabulary is used in either WordEmbeddingInput or softmax classification
    vocab = get_truncated_vocab(train_data, config.min_count)

    # Ensure index of '<oov>' is 0
    for special_word in ['<oov>', '<bos>', '<eos>', '<pad>']:
        if special_word not in word_lexicon:
            word_lexicon[special_word] = len(word_lexicon)

    for word, _ in vocab:
        if word not in word_lexicon:
            word_lexicon[word] = len(word_lexicon)

    # Word Embedding
    if config.word_dim > 0:
        word_embedder = WordEmbed(config.embed_file, config.word_dim, word_lexicon, fix_emb=False)
        logging.info('Word embedding size: {0}'.format(len(word_embedder.word2id)))
    else:
        word_embedder = None
        logging.info('Vocabulary size: {0}'.format(len(word_lexicon)))

    # Character Lexicon
    if config.char_dim> 0:
        char_lexicon = {}
        for sentence in train_data:
            for word in sentence:
                for ch in word:
                    if ch not in char_lexicon:
                        char_lexicon[ch] = len(char_lexicon)

        for special_char in ['<bos>', '<eos>', '<oov>', '<pad>', '<bow>', '<eow>']:
            if special_char not in char_lexicon:
                char_lexicon[special_char] = len(char_lexicon)

        char_embedder = CharEmbed(config.char_dim, char_lexicon, fix_emb=False)
        logging.info('Char embedding size: {0}'.format(len(char_embedder.word2id)))
    else:
        char_lexicon = None
        char_embedder = None

    train = gen_mini_batches(
        train_data, config.batch_size, word_lexicon, char_lexicon,config.max_char_token)

    if config.eval_steps is None:
        config.eval_steps = len(train[0])
    # logging.info('Evaluate every {0} batches.'.format(config.eval_steps))

    valid = None
    test = None

    # if valid_data is not None:
    #   valid = create_batches(
    #     valid_data, config.batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, use_cuda=use_cuda)
    # else:
    #   valid = None

    # if test_data is not None:
    #   test = create_batches(
    #     test_data, config.batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, use_cuda=use_cuda)
    # else:
    #   test = None

    label_to_ix = word_lexicon
    logging.info('vocab size: {0}'.format(len(label_to_ix)))

    nclasses = len(label_to_ix)

    # for s in train_w:
    #  for i in range(s.view(-1).size(0)):
    #    if s.view(-1)[i] >= nclasses:
    #      print(s.view(-1)[i])

    model = Model(word_embedder=word_embedder, char_embedder=char_embedder, n_class=nclasses,
                  encode=config.encode, char_dim=config.char_dim,filters=config.filters,
                  n_highway=config.n_highway, encode_layers=config.encode_layers,
                  activation=config.activation, encode_dim=config.encode_dim,
                  cell_clip=config.cell_clip, proj_clip=config.proj_clip,
                  projection_dim=config.projection_dim, dropout=config.dropout, use_cuda=use_cuda)
    logging.info(str(model))
    if use_cuda:
        model = model.cuda()

    need_grad = lambda x: x.requires_grad
    # optimizer = optim.SGD(filter(need_grad, model.parameters()), lr=config.lr)

    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=config.lr)
    elif config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(filter(need_grad, model.parameters()), lr=config.lr)
    elif config.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(filter(need_grad, model.parameters()), lr=config.lr)
    else:
        raise ValueError('Unknown configimizer {}'.format(config.configimizer.lower()))

    try:
        os.makedirs(config.model_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    if elmo_config['token_embedder']['char_dim'] > 0:
        with codecs.open(os.path.join(config.model_path, 'char.dic'), 'w', encoding='utf-8') as fpo:
            for ch, i in char_embedder.word2id.items():
                print('{0}\t{1}'.format(ch, i), file=fpo)

    with codecs.open(os.path.join(config.model_path, 'word.dic'), 'w', encoding='utf-8') as fpo:
        for w, i in word_lexicon.items():
            print('{0}\t{1}'.format(w, i), file=fpo)

    json.dump(vars(config), codecs.open(os.path.join(config.model_path, 'config.json'), 'w', encoding='utf-8'))

    best_train = 1e+8

    for epoch in range(config.max_epoch):
        best_train = train_epoch(epoch, config, model, optimizer, train, best_train)
        if config.lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= config.lr_decay

    logging.info("best train ppl: {:.6f}.".format(best_train))


if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == 'train':
    train()
