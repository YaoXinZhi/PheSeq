# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 14/10/2021 17:32
@Author: XINZHI YAO
"""

"""
Dynamic Meta Embedding code of self version.
"""

import os
import math

import torch
import torch.nn as nn

import torch.nn.functional as F

def init_weight(weight, method):
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')


def nn_init(nn_module, method='xavier'):
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)


class SingleEmbedder(nn.Module):
    # load single version of embedding.
    def __init__(self, args, entrez_to_idx, embedding_type):
        super(SingleEmbedder, self).__init__()
        print(f'{embedding_type} embedder initializing.')
        self.args = args
        self.entrez_to_idx = entrez_to_idx

        self.embedder = nn.Embedding(args.entrez_size, args.emb_size)
        self.embedding_type = embedding_type

        self.bound = math.sqrt(3.0 / args.emb_size)
        self.embedder.weight.data.uniform_(-1.0 * self.bound, self.bound)
        # print('original')
        # print(self.embedder.weight.data)
        # read different version of embedding
        # return a embedding dictionary
        # self.entrez_set
        pretrained_embeddings = self.load_embedding()

        # print('pretrained')
        # print(pretrained_embeddings)
        # update self.embedder
        indices_matched, indices_missed = self.math_pretrained_embeddings(self.embedder.weight.data, pretrained_embeddings)
        # print('matched')
        # print(self.embedder.weight.data)

        self.embedder.weight.data.copy_(self.normalize_embeddings(self.embedder.weight.data, indices_matched, indices_missed))
        # print('normalizaed')
        # print(self.embedder.weight.data)
        # self.embedder.cuda()
        self.embedder.weight.requires_grad = False

    def forward(self, x):
        return self.embedder(x)

    @staticmethod
    def read_embedding(embedding_file: str,):
        # print('Reading embedding file.')
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

        save_entrez_set = self.entrez_to_idx.keys()

        embedding_file = f'{self.args.embedding_path}/entrez.{self.embedding_type}.embedding.txt'

        entrez_to_embedding = self.read_embedding(embedding_file, save_entrez_set)
        # return a dictionary map entrez to embedding
        # 故意让一个嵌入没有
        # entrez_set = {'1222', '1666'}
        # pretrained_embedding_matrix 是一个 entrez2embedding 的矩阵
        # print(pretrained_embedding_matrix.shape)
        # entrez_to_embedding = {}
        #
        # entrez_to_embedding['1222'] = torch.randn(1, self.args.emb_size).squeeze(0)
        # entrez_to_embedding['1666'] = torch.randn(1, self.args.emb_size).squeeze(0)
        # for idx, embedding in zip({'1222', '1666', '1444'}, pretrained_embedding_matrix):
        #     # print(idx, embedding)
        #     entrez_to_embedding[idx] = embedding

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

class CatEmbedder(nn.Module):
    def __init__(self, args, entrez_to_idx):
        super(CatEmbedder, self).__init__()
        assert args.attnnet == 'none'
        self.args = args
        self.entrez_to_idx = entrez_to_idx
        # embedding_type
        self.n_emb = len(args.embedding_type)
        # self.emb_sz = sum([x[2] for x in args.embeds])
        self.emb_sz = self.n_emb * args.emb_size
        # self.emb_names = sorted([name for name, _, _ in args.embeds])
        self.emb_names = sorted(args.embedding_type)
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
        # embedding_type
        self.embedding_type = args.embedding_type
        self.n_emb = len(args.embedding_type)
        assert not (self.n_emb == 1 and args.attnnet != 'none')
        self.emb_sz = args.proj_embed_sz
        self.emb_names = sorted(args.embedding_type)
        self.embedders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        for embedding_type in args.embedding_type:
            self.embedders.update({embedding_type: SingleEmbedder(args, entrez_to_idx, embedding_type)})
            self.projectors.update({embedding_type: nn.Linear(args.emb_size, args.proj_embed_sz)})
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
        # projected = [ self.projectors[name](self.embedders[name](entrez)) for name in self.emb_names ]
        idx = torch.tensor(self.entrez_to_idx[entrez])
        projected = [ self.projectors[name](self.embedders[name](idx)) for name in self.emb_names ]
        if self.args.attnnet == 'none':
            out = sum(projected)
        else:
            projected_cat = torch.cat([p.unsqueeze(2) for p in projected], 2)
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

            attended = projected_cat * self.m_attn.view(s_len, b_size, self.n_emb, 1).expand_as(projected_cat)
            out = attended.sum(2)

        if self.args.nonlin == 'relu':
            out = F.relu(out)
        if self.args.emb_dropout > 0.0:
            out = self.dropout(out)
        return out

class config:
    def __init__(self):

        # {'cat', 'proj_sum'}
        self.mixmode = 'cat'


        self.embedding_path = '../data/DME_embedding'

        self.embedding_type = ['bag', 'graph', 'p-value', 'term']
        self.embedding_type_size = len(self.embedding_type)

        self.entrez_size = 4601
        self.emb_size = 1024

        # none, relu
        self.nonlin = 'none'
        self.emb_dropout = 0.2

        # ['none', 'no_dep_softmax', 'dep_softmax', 'no_dep_gating', 'dep_gating']
        # self.attnnet = 'none'
        self.attnnet = 'no_dep_softmax'

        # ProjSumEmbedder parameters
        self.proj_embed_sz = 20


def get_embedder(args, entrez_to_idx):
    assert args.mixmode in {'cat', 'proj_sum'}
    if args.mixmode == 'cat':
        return CatEmbedder(args, entrez_to_idx)
    elif args.mixmode == 'proj_sum':
        return ProjSumEmbedder(args, entrez_to_idx)

def read_entrez_idx_file(entrez_to_idx_file: str):

    entrez_to_idx = {}
    with open(entrez_to_idx_file) as f:
        f.readline()
        for line in f:
            entrez, idx = line.strip().split('\t')
            idx = int(idx)

            entrez_to_idx[entrez] = idx
    return entrez_to_idx


def main():

    args = config()

    entrez_to_idx_file = '../data/DME_embedding/entrez_to_idx.tsv'

    entrez_to_idx = read_entrez_idx_file(entrez_to_idx_file)

    # entrez_to_idx = {'1222': 0,
    #                  '1333': 1,
    #                  '1666': 2}

    args.entrez_size = len(entrez_to_idx.keys())


    # cat embedder right
    emb_cat = CatEmbedder(args, entrez_to_idx)

    emb_proj = ProjSumEmbedder(args, entrez_to_idx)

    # self.embedder = get_embedder(args, logger)

    # how to use:
    emb = self.embedder(words)


if __name__ == '__main__':
    pass


