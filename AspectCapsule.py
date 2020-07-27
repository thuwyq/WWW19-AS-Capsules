# coding: utf-8
#
# Copyright 2018 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable

from DynamicRNN import DynamicRNN
from CapNet import CapNet


class AspectCapsule(nn.Module):
    '''
    Decoding the sentences in feedbacks
    Inout: sentences
    Output: sentence vectors, feedback vector
    '''
    def __init__(self,
            dim_input,
            dim_hidden,
            n_layers,
            n_labels,
            n_aspects,
            n_vocab,
            embed_list,
            embed_dropout_rate,
            cell_dropout_rate,
            final_dropout_rate,
            bidirectional,
            rnn_type,
            use_cuda):
        super(AspectCapsule, self).__init__()
        self.n_layers = n_layers
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.use_cuda = use_cuda

        self.add_module('embed', nn.Embedding(n_vocab, dim_input))
        self.add_module('embed_dropout', nn.Dropout(embed_dropout_rate))
        self.add_module('rnn_sen', DynamicRNN(dim_input, dim_hidden, n_layers, dropout=(cell_dropout_rate if n_layers > 1 else 0), 
                                            bidirectional=bidirectional, rnn_type=rnn_type, use_cuda=use_cuda))

        self.add_module('cap_net', CapNet(n_aspects, n_labels, n_layers, dim_hidden, bidirectional, final_dropout_rate, self.use_cuda))

        self.init_weights(embed_list)
        ignored_params = list(map(id, self.embed.parameters()))
        self.base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))

    def init_weights(self, embed_list):
        self.embed.weight.data.copy_(torch.from_numpy(embed_list))

    def forward(self, dict_inst):
        embed_input = self.embed(dict_inst['sens'])
        embedded = self.embed_dropout(embed_input)
        length_sen = dict_inst['len_sen']

        output_pad, hidden_encoder = self.rnn_sen(embedded, lengths=length_sen, flag_ranked=False)
        prob_asp, prob_sentiment, attn_asp, attn_sen = self.cap_net(output_pad, length_sen)

        return prob_asp, prob_sentiment, attn_asp, attn_sen
