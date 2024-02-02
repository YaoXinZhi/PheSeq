# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 15/10/2021 9:26
@Author: XINZHI YAO
"""

import os
import copy

import argparse

parser = argparse.ArgumentParser()

# data parameters
parser.add_argument('-ei', dest='entrez_to_idx_file', default='../data/DME_embedding/entrez_to_idx.tsv',
                    help='default: ../data/DME_embedding/entrez_to_idx.tsv')

parser.add_argument('-ep', dest='entrez_p_file', default='../data/AD_GWAS_data/bedtools_data/entrez_p_sentence.tsv',
                    help='default: ../data/AD_GWAS_data/bedtools_data/entrez_p_sentence.tsv')

parser.add_argument('-es', dest='embedding_save_path', default='../data/DME_embedding',
                    help='default: ../data/DME_embedding')

parser.add_argument('-st', dest='significant_threshold',
                    type=float, default=0.05,
                    help='default: 0.05')

# DME parameters
parser.add_argument('-mm', dest='mixmode', default='cat',
                    choices=['cat', 'proj_sum'],
                    help='mixmode, choices: ["cat", "proj_sum"]')

parser.add_argument('-os', dest='original_embed_sz', type=int, default=1024,
                    help='default: 1024')

parser.add_argument('-nl', dest='nonlin', default='none',
                    choices=['none', 'rely'],
                    help='nonlin, choices: ["none", "rely"]')

parser.add_argument('-ed', dest='emb_dropout', type=float, default=0.0,
                    help='default: 0.0')

parser.add_argument('-an', dest='attnnet', default='none',
                    choices=['none', 'no_dep_softmax', 'dep_softmax', 'no_dep_gating', 'dep_gating'],
                    help='attnnet, choices: ["none", "no_dep_softmax", "dep_softmax", "no_dep_gating", "dep_gating"]')

parser.add_argument('-ps', dest='proj_embed_sz', type=int, default=128,
                    help='default: 128')

# model parameters
parser.add_argument('-hs', dest='hidden_size', type=int, default=256,
                    help='default 256.')

# train parameters
parser.add_argument('-tt', dest='train_time', type=int, default=100,
                    help='default: 100')


parser.add_argument('-lr', dest='phi_learning_rate',
                    type=float, default=5e-4,
                    help='default: 5e-4')

parser.add_argument('-bt', dest='batch_train_bool', action='store_true',
                    default=False)

parser.add_argument('-bs', dest='batch_size', type=int, default=25,
                    help='default: 25')

parser.add_argument('-rs', dest='random_seed', type=int, default=126,
                    help='default: 126')

parser.add_argument('-ga', dest='gradient_accumulation', action='store_false',
                    default=False,
                    help='gradient_accumulation, default=True')

parser.add_argument('-sl', dest='save_log', action='store_true',
                    default=False,
                    help='save_log, default: False')

# log parameters
parser.add_argument('-lp', dest='log_save_path', default='../log',
                    help='default ../log')
parser.add_argument('-lf', dest='log_prefix', default='test',
                    help='test')

# trick parameters
parser.add_argument('-uf', dest='use_fake_data', action='store_true',
                    default=False,
                    help='use_fake_data, default: False')

parser.add_argument('-ba', dest='use_best_alpha', action='store_true',
                    default=False,
                    help='use_best_alpha, default: False')




raw_args = parser.parse_known_args()[0]
args = copy.deepcopy(raw_args)


def read_entrez_idx_file(entrez_to_idx_file: str):

    entrez_to_idx = {}
    with open(entrez_to_idx_file) as f:
        f.readline()
        for line in f:
            entrez, idx = line.strip().split('\t')
            idx = int(idx)

            entrez_to_idx[entrez] = idx
    return entrez_to_idx

def get_available_embeddings(embedding_save_path: str):
    """
    entrez.bag.embedding.txt      entrez.term.embedding.txt
entrez.graph.embedding.txt    entrez_to_idx.tsv
entrez.p-value.embedding.txt
    """
    file_list = os.listdir(embedding_save_path)

    available_embedding_set = set()
    for _file in file_list:
        if _file.startswith('entrez') and _file.endswith('.embedding.txt'):
            available_embedding_set.add(_file.split('.')[1])
    return available_embedding_set


def preprocess(opt):
    # log path
    if not os.path.exists(opt.log_save_path):
        os.mkdir(opt.log_save_path)

    # entrez
    opt.entrez_size = 4601

    opt.entrez_to_idx = read_entrez_idx_file(opt.entrez_to_idx_file)

    # embedding
    opt.embedding_type_list = ['bag', 'graph', 'p-value', 'term']
    opt.embedding_type_size = len(opt.embedding_type_list)

    if opt.mixmode == 'cat':
        opt.final_embedding_size = opt.embedding_type_size * opt.original_embed_sz
    elif opt.mixmode == 'proj_sum':
        opt.final_embedding_size = opt.proj_embed_sz

    # embedding version
    opt.available_embedding = get_available_embeddings(opt.embedding_save_path)

    # check needed embedding_type is available.
    for embedding_type in opt.embedding_type_list:
        if embedding_type not in opt.available_embedding:
            raise ValueError(f'{embedding_type} is not available.')

preprocess(args)

