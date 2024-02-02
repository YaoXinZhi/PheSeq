# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 15/10/2021 9:54
@Author: XINZHI YAO
"""

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

def get_embedder(args):

    assert args.mixmode in {'cat', 'proj_sum'}
    if args.mixmode == 'cat':
        return CatEmbedder(args, args.entrez_to_idx)
    elif args.mixmode == 'proj_sum':
        return ProjSumEmbedder(args, args.entrez_to_idx)


class SingleEmbedder(nn.Module):
    # load single version of embedding.
    def __init__(self, args, entrez_to_idx, embedding_type):
        super(SingleEmbedder, self).__init__()
        print(f'{embedding_type} embedder initializing.')
        self.args = args
        self.entrez_to_idx = entrez_to_idx

        self.embedder = nn.Embedding(args.entrez_size, args.original_embed_sz)
        self.embedding_type = embedding_type

        self.bound = math.sqrt(3.0 / args.original_embed_sz)
        self.embedder.weight.data.uniform_(-1.0 * self.bound, self.bound)

        pretrained_embeddings = self.load_embedding()

        indices_matched, indices_missed = self.math_pretrained_embeddings(self.embedder.weight.data, pretrained_embeddings)

        self.embedder.weight.data.copy_(self.normalize_embeddings(self.embedder.weight.data, indices_matched, indices_missed))
        self.embedder.weight.requires_grad = False

    @staticmethod
    def read_embedding(embedding_file: str,):
        embedding_list = [ ]
        entrez_list = [ ]
        term_embedding_dict = {}
        with open(embedding_file) as f:
            for line in f:
                l = line.strip().split('\t')
                if len(l) != 2:
                    print(l)
                    input()

                term = '-'.join(l[ 0 ].split('-')[ 1: ])

                entrez_list.append(l[ 0 ])

                embedding = list(map(float, l[ 1 ].split()))
                embedding_list.append(embedding)
                term_embedding_dict[ term ] = embedding

        return term_embedding_dict

    # todo: load read data.
    def load_embedding(self):

        embedding_file = f'{self.args.embedding_save_path}/entrez.{self.embedding_type}.embedding.txt'
        entrez_to_embedding = self.read_embedding(embedding_file)

        return entrez_to_embedding

    def math_pretrained_embeddings(self, embeddings, pretrained_embeddings):
        # copy pretrained_embeddings to self.embedder
        indices_matched, indices_missed = [], []

        entrez_set = set(pretrained_embeddings.keys())
        for entrez, idx in self.entrez_to_idx.items():
            # this embedding appeared in pretrained embeddings.
            if entrez in entrez_set:
                embeddings[idx].copy_(pretrained_embeddings[entrez])
                indices_matched.append(idx)
            else:
                indices_missed.append(idx)
        return indices_matched, indices_missed

    @staticmethod
    def normalize_embeddings(embeddings, indices_to_normalize, indices_to_zero):
        if len(indices_to_normalize) > 0:
            embeddings = embeddings - embeddings[ indices_to_normalize, : ].mean(0)
        if len(indices_to_zero) > 0:
            embeddings[ indices_to_zero, : ] = 0
        return embeddings

    def forward(self, x):
        return self.embedder(x)


class CatEmbedder(nn.Module):
    def __init__(self, args, entrez_to_idx):
        super(CatEmbedder, self).__init__()
        assert args.attnnet == 'none'
        self.args = args
        self.entrez_to_idx = entrez_to_idx
        # embedding_type
        self.n_emb = len(args.embedding_type_list)

        self.emb_sz = self.n_emb * args.original_embed_sz
        self.emb_names = sorted(args.embedding_type_list)

        self.embedders = nn.ModuleDict({
            embedding_type: SingleEmbedder(args, entrez_to_idx, embedding_type) for embedding_type in self.emb_names
        })
        self.dropout = nn.Dropout(p=args.emb_dropout)

    def forward(self, entrez):
        idx = self.entrez_to_idx[entrez]
        out = torch.cat([self.embedders[embedding_type](torch.tensor(idx)) for embedding_type in self.emb_names], -1)


        if self.args.nonlin == 'relu':
            out = F.relu(out)
        if self.args.emb_dropout > 0.0:
            out = self.dropout(out)
        return out


class ProjSumEmbedder(nn.Module):
    def __init__(self, args, entrez_to_idx):
        super(ProjSumEmbedder, self).__init__()
        assert args.attnnet in {'none', 'no_dep_softmax', 'dep_softmax', 'no_dep_gating', 'dep_gating'}
        self.args = args
        self.entrez_to_idx = entrez_to_idx
        self.embedding_type_list = args.embedding_type_list
        self.n_emb = len(args.embedding_type_list)
        assert not (self.n_emb == 1 and args.attnnet != 'none')
        self.emb_sz = args.proj_embed_sz
        self.emb_names = sorted(args.embedding_type_list)
        self.embedders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        for embedding_type in args.embedding_type_list:
            self.embedders.update({embedding_type: SingleEmbedder(args, entrez_to_idx, embedding_type)})
            self.projectors.update({embedding_type: nn.Linear(args.original_embed_sz, args.proj_embed_sz)})
            nn_init(self.projectors[embedding_type], 'xavier')

        self.attn_0, self.attn_1 = None, None
        self.m_attn = None
        if self.n_emb > 1 and args.attnnet != 'none':
            if args.attnnet.startswith('dep_'):
                self.attn_0 = nn.LSTM(args.proj_embed_sz, 2, bidirectional=True)
                nn_init(self.attn_0, 'orthogonal')
                self.attn_1 = nn.Linear(2 * 2, 1)
                nn_init(self.attn_1, 'xavier')

            elif args.attnnet.startswith('no_dep_'):
                self.attn_0 = nn.Linear(args.proj_embed_sz, 2)
                nn_init(self.attn_0, 'xavier')
                self.attn_1 = nn.Linear(2, 1)
                nn_init(self.attn_1, 'xavier')

        self.dropout = nn.Dropout(p=args.emb_dropout)

    def forward(self, entrez):
        idx = torch.tensor(self.entrez_to_idx[entrez])
        projected = [ self.projectors[name](self.embedders[name](idx)) for name in self.emb_names ]

        if self.args.attnnet == 'none':
            out = sum(projected)
        else:
            # print(projected)
            # print(projected[0].shape)
            # 3 * 128
            # projected_cat = torch.cat([p.unsqueeze(2) for p in projected], 2)
            projected_cat = torch.cat([p.reshape(1,1,1,-1) for p in projected], 2)
            s_len, b_size, _, emb_dim = projected_cat.size()
            attn_input = projected_cat

            if self.args.attnnet.startswith('dep_'):
                attn_input = attn_input.view(s_len, b_size * self.n_emb, -1)
                self.m_attn = self.attn_1(self.attn_0(attn_input)[0])
                self.m_attn = self.m_attn.view(s_len, b_size, self.n_emb)

            elif self.args.attnnet.startswith('no_dep_'):
                self.m_attn = self.attn_1(self.attn_0(attn_input)).squeeze(3)

            if self.args.attnnet.endswith('_gating'):
                self.m_attn = torch.sigmoid(self.m_attn)
            elif self.args.attnnet.endswith('_softmax'):
                self.m_attn = F.softmax(self.m_attn, dim=2)
            # print(f'm_attn: {self.m_attn.shape}')
            attended = projected_cat * self.m_attn.view(s_len, b_size, self.n_emb, 1).expand_as(projected_cat)

            out = attended.sum(2)
            # print(f'out: {out.shape}')
        # print(out.shape)
        if self.args.nonlin == 'relu':
            out = F.relu(out)
        if self.args.emb_dropout > 0.0:
            out = self.dropout(out)
        return out

