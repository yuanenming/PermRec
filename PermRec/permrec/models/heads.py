'''
Description: 
Author: Enming Yuan
email: yem19@mails.tsinghua.edu.cn
Date: 2021-08-15 20:59:00
LastEditTime: 2021-08-21 16:35:36
'''

import torch
import torch.nn as nn


class LinearPredictionHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vocab_size = args.num_items + 1
        hidden = args.d_model
        self.out = nn.Linear(hidden, self.vocab_size)

    def forward(self, x, candidates=None):
        x = self.out(x)  # B x V or M x V
        if candidates is not None:
            x = x.gather(1, candidates)  # B x C or M x C
        return x


class DotProductPredictionHead(nn.Module):
    """share embedding parameters"""
    def __init__(self, d_model, num_items, token_embeddings):
        super().__init__()
        self.token_embeddings = token_embeddings
        self.vocab_size = num_items + 1
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))

    def forward(self, x, candidates=None):
        if candidates is not None:  # x : B x H
            emb = self.token_embeddings(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
            bias = self.bias.expand(logits.size(0), -1).gather(1, candidates)  # B x C
            logits += bias
        else:  # x : M x H
            emb = self.token_embeddings.weight[:self.vocab_size]  # V x H
            logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
            logits += self.bias
        return logits