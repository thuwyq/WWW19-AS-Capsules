# coding: utf-8
#
# Copyright 2018 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#

from __future__ import unicode_literals, print_function, division

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from DynamicRNN import DynamicRNN
from attention import AttentionPair


class CapNet(nn.Module):
    def __init__(self, n_cap, n_label, n_layers, dim_hidden, flag_bid, dropout_rate=0, use_cuda=False):
        super(CapNet, self).__init__()
        self.n_cap = n_cap
        self.use_cuda = use_cuda
        self.add_module('rnn', DynamicRNN(dim_hidden + dim_hidden * (2 if flag_bid else 1), dim_hidden, n_layers, dropout=0.5 if n_layers > 1 else 0, 
                                            bidirectional=flag_bid, rnn_type='LSTM', use_cuda=use_cuda))
        for i in range(n_cap):
            self.add_module('cap_%s' % i, Capsule(dim_hidden, flag_bid, n_label, dropout_rate, use_cuda))

    def forward(self, output_pad, lengths):
        list_prob, list_prob_s, list_alpha_asp, list_alpha_sen = [], [], [], []
        for i in range(self.n_cap):
            prob_tmp, prob_sentiemnt_tmp, alpha_asp, alpha_sen = getattr(self, 'cap_%s' % i)(output_pad, lengths, self.rnn)
            list_prob.append(prob_tmp)
            list_prob_s.append(prob_sentiemnt_tmp)
            list_alpha_asp.append(alpha_asp)
            list_alpha_sen.append(alpha_sen)

        prob = torch.transpose(torch.stack(list_prob).squeeze(-1), 0, 1)
        prob_sentiment = torch.transpose(torch.stack(list_prob_s).squeeze(-1), 0, 1)

        list_alpha_asp = torch.stack(list_alpha_asp)
        output_alpha_asp = []
        for i in range(len(lengths)):
            output_alpha_asp.append(list_alpha_asp[:, i, :])
        output_alpha_asp = torch.stack(output_alpha_asp)

        list_alpha_sen = torch.stack(list_alpha_sen)
        output_alpha_sen = []
        for i in range(len(lengths)):
            output_alpha_sen.append(list_alpha_sen[:, i, :])
        output_alpha_sen = torch.stack(output_alpha_sen)

        return prob, prob_sentiment, output_alpha_asp, output_alpha_sen


class Capsule(nn.Module):
    def __init__(self, dim_hidden, flag_bid, n_label, dropout_rate, use_cuda=True):
        super(Capsule, self).__init__()
        dim_rep = dim_hidden * (2 if flag_bid else 1)
        self.add_module('linear_asp', nn.Linear(dim_rep * 3, 1))
        self.add_module('linear_sen', nn.Linear(dim_rep * 3, n_label))
        self.add_module('final_dropout', nn.Dropout(dropout_rate))
        
        self.add_module('attn_pair_sen', AttentionPair(dim_hidden, dim_hidden, flag_bid=flag_bid, use_cuda=use_cuda))
        self.add_module('attn_pair_asp', AttentionPair(dim_hidden, dim_hidden, flag_bid=flag_bid, use_cuda=use_cuda))
        self.add_module('attn_shared', AttentionPair(dim_hidden, dim_hidden, flag_bid=flag_bid, use_cuda=use_cuda))
        self.v_kernel = Parameter(torch.FloatTensor(1, dim_hidden))
        self.reset_parameters(dim_hidden)

    def reset_parameters(self, dim_hidden):
        stdv = 1.0 / math.sqrt(dim_hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, matrix_hidden_pad, len_hidden_pad, rnn):
        h_Matrix = torch.cat((matrix_hidden_pad, self.v_kernel.repeat(matrix_hidden_pad.size(0), matrix_hidden_pad.size(1), 1)), dim=-1)
        output_pad, _ = rnn(h_Matrix, lengths=len_hidden_pad, flag_ranked=False)

        # feature shared
        feature_shared, _ = self.attn_shared(self.v_kernel, output_pad, len_hidden_pad)

        # tackle sentiment
        r_s_sen, attn_sen = self.attn_pair_sen(self.v_kernel, output_pad, len_hidden_pad)
        feature_attn_sen = torch.mul(matrix_hidden_pad, attn_sen.unsqueeze(-1).expand_as(matrix_hidden_pad)).sum(dim=1)
        prob_sentiment = torch.softmax(self.linear_sen(self.final_dropout(torch.cat([r_s_sen, feature_attn_sen, feature_shared], dim=-1))), dim=-1)

        # tackle aspect
        r_s_asp, attn_asp = self.attn_pair_asp(self.v_kernel, output_pad, len_hidden_pad)
        feature_attn_asp = torch.mul(matrix_hidden_pad, attn_asp.unsqueeze(-1).expand_as(matrix_hidden_pad)).sum(dim=1)
        prob_asp = torch.sigmoid(self.linear_asp(self.final_dropout(torch.cat([r_s_asp, feature_attn_asp, feature_shared], dim=-1))))
        
        return prob_asp, prob_sentiment, attn_asp, attn_sen
