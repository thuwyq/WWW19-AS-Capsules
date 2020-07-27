# coding: utf-8
#
# Copyright 2018 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#

from __future__ import unicode_literals, print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from AspectCapsule import AspectCapsule


class Lang:
    def __init__(self, vocab):
        self.index2word = {}
        self.word2index = {}
        for i in range(len(vocab)):
            self.index2word[i] = vocab[i]
            self.word2index[vocab[i]] = i

    def indexFromSentence(self, sentence, flag_list=True):
        list_ = sentence if flag_list else sentence.lower().split()
        list_idx = []
        for word in list_:
            list_idx.append(self.word2index[word] if word in self.word2index else self.word2index['<unk>'])
        return list_idx

    def VariablesFromSentences(self, sentences, flag_list=True, use_cuda=True):
        '''
        if sentence is a list of word, flag_list should be True in the training 
        '''
        indexes = [self.indexFromSentence(sen, flag_list) for sen in sentences]
        inputs = Variable(torch.LongTensor(indexes))
        return inputs.cuda() if use_cuda else inputs


class bridgeModel(nn.Module):
    def __init__(self,
            dim_input,
            dim_hidden,
            n_layers,
            n_labels,
            n_aspects,
            batch_size,
            max_length_sen,
            learning_rate,
            lr_word_vector=0.01,
            weight_decay=0,
            vocab=None,
            embed=None,
            embed_dropout_rate=0.,
            cell_dropout_rate=0.,
            final_dropout_rate=0.,
            bidirectional=True,
            optim_type="Adam",
            rnn_type="LSTM",
            lambda1=0.001,
            use_cuda=True):
        super(bridgeModel, self).__init__()
        self.max_length_sen = max_length_sen
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda1 = lambda1
        self.use_cuda = use_cuda
        self.lang = Lang(vocab) 
        self.model = AspectCapsule(dim_input, dim_hidden, n_layers, n_labels, n_aspects, len(vocab), embed,
                embed_dropout_rate, cell_dropout_rate, final_dropout_rate, bidirectional, rnn_type, use_cuda)
        if self.use_cuda:
            self.model.cuda()
        self.optimizer = getattr(optim, optim_type)([
                                        {'params': self.model.base_params, 'weight_decay': weight_decay},
                                        {'params': self.model.embed.parameters(), 'lr': lr_word_vector, 'weight_decay': 0}
                                    ], lr=self.learning_rate)

    def get_batch_data(self, batched_data):
        dict_data = {}

        dict_data['sens'] = self.lang.VariablesFromSentences(batched_data['batch_sentences'], True, self.use_cuda)
        dict_data['len_sen'] = batched_data['batch_length_sen']

        labels_asp = Variable(torch.FloatTensor(batched_data['batch_labels_asp']))
        dict_data['labels_asp'] = labels_asp.cuda() if self.use_cuda else labels_asp

        labels_sent = Variable(torch.LongTensor(batched_data['batch_labels']))
        dict_data['labels_sent'] = labels_sent.cuda() if self.use_cuda else labels_sent

        return dict_data

    def stepTrain(self, batched_data, inference=False):
        # Turn on training mode which enables dropout.
        self.model.eval() if inference else self.model.train()
        
        if inference == False:
            # zero the parameter gradients
            self.optimizer.zero_grad()

        b_data = self.get_batch_data(batched_data)
        prob_asp, prob_sentiment, attn_asp, attn_sen = self.model(b_data)
        loss_asp = F.binary_cross_entropy(prob_asp, b_data['labels_asp'])
        # cross entropy in sentiment classification
        loss_sent = F.nll_loss(torch.log(prob_sentiment.permute(0, 2, 1)), b_data['labels_sent'], ignore_index=-1)
        loss = self.lambda1 * loss_asp + (1 - self.lambda1) * loss_sent

        if inference == False:
            loss.backward()
            self.optimizer.step()
        
        return np.array([loss.data.cpu().numpy(), loss_asp.data.cpu().numpy(), loss_sent.data.cpu().numpy()]).reshape(3), \
                    prob_asp.data.cpu().numpy(), prob_sentiment.data.cpu().numpy(), \
                    attn_asp.data.cpu().numpy(), attn_sen.data.cpu().numpy()

    def save_model(self, dir, idx):
        os.mkdir(dir) if not os.path.isdir(dir) else None
        torch.save(self.state_dict(), '%s/model%s.pth' % (dir, idx))
