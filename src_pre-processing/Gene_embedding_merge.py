# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 24/09/2021 15:14
@Author: XINZHI YAO
"""

import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import manifold, datasets

def read_entrez_p(entrez_p_file: str):
    entrez_to_p = {}
    entrez_to_symbol = {}
    entrez_bag = defaultdict(list)
    entrez_tag = defaultdict(list)

    entrez_to_tag_set = defaultdict(set)
    with open(entrez_p_file) as f:
        for line in f:
            l = line.strip().split('\t')
            if line.startswith('GENE_LINE'):
                symbol, entrez, p = l[ 1 ], l[ 2 ], float(l[ -1 ])
                entrez_to_p[ entrez ] = p
                entrez_to_symbol[ entrez ] = symbol
            else:
                pmid = l[ 0 ]
                sentence = l[ 1 ]
                tags = eval(l[ 2 ])

                # print(tags)
                entrez_to_tag_set[entrez].update(tags)

                entrez_bag[entrez].append((pmid, sentence))
                entrez_tag[entrez].append(tags)
    print(f'data size: {len(entrez_to_p)}, \
            min_p: {min(entrez_to_p.values())}, \
            max_p: {max(entrez_to_p.values())}.')
    return entrez_to_p, entrez_to_tag_set

# read embedding file
def read_embedding(embedding_file: str, save_term=None, split_term=True):
    embedding_list = [ ]
    entrez_list = [ ]
    term_to_embedding = {}
    with open(embedding_file) as f:
        for line in f:
            l = line.strip().split('\t')
            if len(l) != 2:
                print(l)
                input()

            if split_term:
                term = '-'.join(l[0].split('-')[1:])
            else:
                term = l[0]

            if not save_term is None:
                if term in save_term:
                    entrez_list.append(l[ 0 ])

                    embedding = list(map(float, l[ 1 ].split()))
                    embedding_list.append(embedding)
                    term_to_embedding[term] = np.array(embedding)
            else:
                entrez_list.append(l[ 0 ])

                embedding = list(map(float, l[ 1 ].split()))
                embedding_list.append(embedding)
                term_to_embedding[ term ] = np.array(embedding)

    print(f'data size: {len(entrez_list):,}, feature size: {len(embedding_list[ 0 ])}')
    return entrez_list, np.array(embedding_list), term_to_embedding


def trend_bag_embedding_merge(entrez_trend_embedding_dict: dict,
                              entrez_bag_embedding_dict: dict,
                              entrez_to_graph_embedding: dict,
                              merge_lambda: float):

    if merge_lambda > 1:
        raise ValueError(f'merge_lambda must be between 0 and 1, got {merge_lambda}.')

    embedding_size = len(list(entrez_to_graph_embedding.values())[0])

    if len(list(entrez_to_graph_embedding.values())[0]) != len(list(entrez_bag_embedding_dict.values())[0]) \
        or len(list(entrez_to_graph_embedding.values())[0]) != len(list(entrez_to_graph_embedding.values())[0]):
        raise ValueError(f'Text/Context/Graph Embedding should have save size.')

    entrez_embedding_dict = {}
    for idx, entrez in enumerate(entrez_trend_embedding_dict.keys()):

        if entrez_to_graph_embedding.get(entrez):
            trend_embedding = entrez_trend_embedding_dict[entrez]
        else:
            trend_embedding = [0] * embedding_size

        if entrez_bag_embedding_dict.get(entrez):
            bag_embedding = entrez_bag_embedding_dict[entrez]
        else:
            bag_embedding = [ 0 ] * embedding_size

        if entrez_to_graph_embedding.get(entrez):
            graph_embedding = entrez_to_graph_embedding[entrez]
        else:
            graph_embedding = [0] * embedding_size

        # average merge
        merge_embedding = (bag_embedding + trend_embedding + graph_embedding) / 3

        entrez_embedding_dict[entrez] = merge_embedding

    return entrez_embedding_dict


def save_embedding(term_embedding_dict: dict, save_file: str):

    with open(save_file, 'w') as wf:
        for term, embedding in term_embedding_dict.items():
            embedding_wf = ' '.join(list(map(str, embedding)))
            wf.write(f'{term}\t{embedding_wf}\n')
    print(f'{save_file} save done.')

def save_plot(plot_save: str, embedding_file: str, entrez_p_dict):
    entrez_list, embedding_list, _ = read_embedding(embedding_file)

    print('TSNE running.')
    start_time = time.time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    embedding_tsne = tsne.fit_transform(embedding_list)
    end_time = time.time()
    print(f'TSNE done, cost: {end_time - start_time:.2f} s.')
    # p-value
    p_list = [ entrez_p_dict[ entrez ] for entrez in entrez_list ]

    plt.figure(figsize=(8, 8))
    plt.scatter(embedding_tsne[ :, 0 ], embedding_tsne[ :, 1 ], marker='o', c=p_list, cmap='summer')

    plt.colorbar()

    plt.savefig(plot_save)
    print(f'{plot_save} save done.')

def save_plot_new(plot_save: str, embedding_file: str, entrez_p_dict):
    entrez_list, embedding_list, _ = read_embedding(embedding_file)

    print('TSNE running.')
    start_time = time.time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    embedding_tsne = tsne.fit_transform(embedding_list)
    end_time = time.time()
    print(f'TSNE done, cost: {end_time - start_time:.2f} s.')

    p_list = [ entrez_p_dict[ entrez ] for entrez in entrez_list ]

    # 根据p值将数据分割为显著的部分和不显著的部分 分别进行可视化
    sig_embedding = [ ]
    sig_p_list = [ ]

    insig_embedding = [ ]
    insig_p_list = [ ]
    for idx, embedding in enumerate(embedding_tsne):
        if p_list[ idx ] <= 0.05:
            sig_embedding.append(embedding)
            sig_p_list.append(p_list[ idx ])
        else:
            insig_embedding.append(embedding)
            insig_p_list.append(p_list[ idx ])
    sig_embedding = np.array(sig_embedding)
    insig_embedding = np.array(insig_embedding)

    print(f'sig_embedding: {sig_embedding.shape}, insig_embedding: {insig_embedding.shape}')

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    images = [ [
        axes.scatter(sig_embedding[ :, 0 ], sig_embedding[ :, 1 ], marker='o', c=sig_p_list, cmap='Oranges', alpha=0.2),

        axes.scatter(insig_embedding[ :, 0 ], insig_embedding[ :, 1 ], marker='o', c=insig_p_list, cmap='summer')
    ] ]

    fig.colorbar(images[ 0 ][ 1 ], ax=axes, fraction=.05, pad=0.15)
    fig.colorbar(images[ 0 ][ 0 ], ax=axes, fraction=.05, pad=0.15)

    fig.savefig(plot_save)
    print(f'{plot_save} save done.')

def read_graph_embedding(graph_embedding_file: str, entrez_set: set):

    entrez_to_graph_embedding = {}
    with open(graph_embedding_file) as f:
        for line in f:
            l = line.strip().split('\t')
            entrez = l[0]
            embedding = list(map(float, l[1].split()))

            if entrez not in entrez_set:
                continue
            entrez_to_graph_embedding[entrez] = embedding
    return entrez_to_graph_embedding


def main():
    """
    该代码用于
    1. 读取Bag对于每个entrez的嵌入。
    2. 读取normal_embedding和MPA/CPA_embedding加权平均得到的嵌入。
        mu_factor = 5, sigma = 0.1
        mu = p * mu_factor
        merge_beta = 0.1
    3. 以Bag嵌入作为base嵌入，进行加权平均。
    4. 保存嵌入。
    5. 可视化。
    """

    parser = argparse.ArgumentParser(description='Tendency bag embedding merge.')

    parser.add_argument('--text_embedding_file', dest='text_embedding_file', required=True)

    parser.add_argument('--concept_embedding_file', dest='concept_embedding_file', required=True)

    parser.add_argument('--graph_embedding_file', required=True)

    parser.add_argument('--embedding_save_file', dest='embedding_save_file', required=True)


    args = parser.parse_args()

    Bag_embedding_file = args.text_embedding_file
    _, bag_embedding_array, entrez_to_bag_embedding = read_embedding(Bag_embedding_file, None, False)
    print(f'bag_embedding: {bag_embedding_array.shape}')
    beta = 0.5

    entrez_set = set(entrez_to_bag_embedding.keys())
    entrez_to_graph_embedding = read_graph_embedding(args.graph_embedding_file, entrez_set)
    print(f'{len(entrez_to_graph_embedding):,} gene in graph_embedding.')

    entrez_trend_embedding_file = args.concept_embedding_file
    entrez_list, trend_embedding_array, entrez_to_trend_embedding = read_embedding(entrez_trend_embedding_file, None, False)
    print(f'trend_embedding: {trend_embedding_array.shape}')


    entrez_to_merge_embedding = trend_bag_embedding_merge(entrez_to_trend_embedding,
                                                          entrez_to_bag_embedding,
                                                          entrez_to_graph_embedding,
                                                          beta)

    embedding_save_file =args.embedding_save_file
    save_embedding(entrez_to_merge_embedding, embedding_save_file)


if __name__ == '__main__':
    main()
