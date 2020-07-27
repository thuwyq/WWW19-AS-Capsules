# coding: utf-8
#
# Copyright 2018 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AttentionPair(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, dim_vect, dim_attn, flag_bid, use_cuda=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(AttentionPair, self).__init__()
        self.use_cuda = use_cuda
        dim_attn_bid = dim_attn * (2 if flag_bid else 1)
        self.add_module('linear_vec', nn.Linear(dim_vect, dim_attn, bias=False))
        self.add_module('linear_mat', nn.Linear(dim_attn_bid, dim_attn, bias=False))
        self.add_module('linear_attn', nn.Linear(dim_attn, 1, bias=False))

    def forward(self, vector, matrix, input_lengths):
        """ Forward pass.
        # Arguments:
            vect (Torch.Variable): Tensor of input vector
            matrix (Torch.Variable): Tensor of input matrix
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            representations and attentions.
        """
        t1 = self.linear_vec(vector)
        t2 = self.linear_mat(matrix)
        t3 = F.relu(t1.unsqueeze(1) + t2)
        logits = self.linear_attn(t3).squeeze(-1)
        unnorm_ai = (logits - logits.max()).exp()

        max_len = torch.max(torch.LongTensor(input_lengths))
        indices = torch.arange(0, max_len).unsqueeze(0)
        mask = Variable((indices < torch.LongTensor(input_lengths).unsqueeze(1)).float())
        mask = mask.cuda() if self.use_cuda else mask

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(matrix, attentions.unsqueeze(-1).expand_as(matrix))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return representations, attentions
