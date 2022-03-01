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
                              merge_lambda: float):

    if merge_lambda > 1:
        raise ValueError(f'merge_lambda must be between 0 and 1, got {merge_lambda}.')

    entrez_embedding_dict = {}
    for idx, entrez in enumerate(entrez_trend_embedding_dict.keys()):

        trend_embedding = entrez_trend_embedding_dict[entrez]

        bag_embedding = entrez_bag_embedding_dict[entrez]
        # if idx == 1:
        #     print(entrez)
        #     print(merge_lambda)
        #     print(bag_embedding[:5])
        #     print(trend_embedding[:5])
        merge_embedding = merge_lambda*bag_embedding + (1-merge_lambda)*trend_embedding
        # if idx == 1:
        #     print(merge_embedding[:5])
        #     input()


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
    parser.add_argument('--summary_data_path', dest='summary_data_path', type=str,
                        default='../data/AD_GWAS_data/other_GWAS/CNCR_data/NG_2019_bedtools',
                        help='default: ../data/AD_GWAS_data/other_GWAS/CNCR_data/NG_2019_bedtools')
    parser.add_argument('--base_embedding_save_path', dest='base_embedding_save_path',
                        default='../result/NG_2019_result',
                        help='default: ../result/NG_2019_result')
    parser.add_argument('--trend_embedding_dir', dest='trend_embedding_dir',
                        default='p_norm_hpo_embedding_distinctive',
                        help='default: p_norm_hpo_embedding_distinctive')
    parser.add_argument('--embedding_save_dir', dest='embedding_save_dir',
                        default='trend_p_hpo_embedding_distinctive',
                        help='default: trend_p_hpo_embedding_distinctive')
    parser.add_argument('--save_plot', dest='save_plot', action='store_true',
                        default=False,
                        help='save_plot, default: False')
    parser.add_argument('--plot_save_dir', dest='plot_save_dir',
                        default='distinctive_plot',
                        help='default: distinctive_plot')

    args = parser.parse_args()

    # 1.
    # change 1
    # summary_data_path = '../data/AD_GWAS_data/other_GWAS/CNCR_data/NG_2019_bedtools'
    # summary_data_path = '../data/AD_GWAS_data/other_GWAS/CNCR_data/Genet_2021_bedtools'

    # change 2
    # base_embedding_save_path = '../result/NG_2019_result'
    # base_embedding_save_path = '../result/Genet_2021_result'

    # change 3
    # trend_embedding_save_path = f'{base_embedding_save_path}/p_norm_hpo_embedding'
    trend_embedding_save_path = f'{args.base_embedding_save_path}/{args.trend_embedding_dir}'

    # change 4
    # merge_embedding_save_path = f'{base_embedding_save_path}/trend_p_hpo_embedding'
    merge_embedding_save_path = f'{args.base_embedding_save_path}/{args.embedding_save_dir}'

    if not os.path.exists(merge_embedding_save_path):
        os.mkdir(merge_embedding_save_path)

    Bag_embedding_file = f'{args.base_embedding_save_path}/entrez_p_sentence.embedding.txt'
    _, bag_embedding_array, entrez_to_bag_embedding = read_embedding(Bag_embedding_file, None, False)
    print(f'bag_embedding: {bag_embedding_array.shape}')

    # loop 寻优
    # 对于三类trend_beta，都生成后续图看看:
    # for normal_beta in {'0.20', '0.25', '0.30', '0.35', '0.40'}:

    # 2. 读取normal_embedding和MPA/CPA_embedding加权平均得到的嵌入。
    # entrez_p_file = f'{args.summary_data_path}/entrez_p_sentence.HPO.tsv'
    # entrez_to_p, entrez_to_tag_set = read_entrez_p(entrez_p_file)

    for normal_beta in range(0, 100, 1):
        normal_beta = normal_beta * 0.1 + 0.00
        if normal_beta > 1:
            break

        entrez_trend_embedding_file = f'{trend_embedding_save_path}/entrez.MPA-CPA.total.{normal_beta:.2f}.embedding.txt'
        entrez_list, trend_embedding_array, entrez_to_trend_embedding = read_embedding(entrez_trend_embedding_file, None, False)
        print(f'trend_embedding: {trend_embedding_array.shape}')

        # 批量储存beta 看最优
        for beta in range(0, 100, 1):
            beta = beta * 0.1 + 0.0
        # for beta in {0.50}:
            if beta > 1:
                break
            print(f'Merge-Beta: {beta:.2f}')

            # 3.
            entrez_to_merge_embedding = trend_bag_embedding_merge(entrez_to_trend_embedding,
                                                                  entrez_to_bag_embedding,
                                                                  beta)
            # 4.
            # embedding_save_file = f'../result/trend_embedding/trend_bag_embedding.TrendBeta-{normal_beta:.2f}.beta-{beta:.2f}.txt'
            # embedding_save_file = f'../result/trend_log_p_vec_embedding/trend_bag_embedding.TrendBeta-{normal_beta:.2f}.beta-{beta:.2f}.txt'
            embedding_save_file = f'{merge_embedding_save_path}/trend_bag_embedding.TrendBeta-{normal_beta:.2f}.beta-{beta:.2f}.txt'
            save_embedding(entrez_to_merge_embedding, embedding_save_file)

            # 5.
            if args.save_plot:
                plot_save_file = f'{args.base_embedding_save_path}/{args.plot_save_dir}/trend_bag_embedding.TrendBeta-{normal_beta:.2f}.beta-{beta:.2f}.png'
                # save_plot(plot_save_file, embedding_save_file, entrez_to_p)
                save_plot_new(plot_save_file, embedding_save_file, entrez_to_p)

    # # 1.
    # Bag_embedding_file = '../result/entrez_p_sentence.embedding.large.norm.txt'
    # _, bag_embedding_array, entrez_to_bag_embedding = read_embedding(Bag_embedding_file, None, False)
    # print(f'bag_embedding: {bag_embedding_array.shape}')

    # # 2. 读取normal_embedding和MPA/CPA_embedding加权平均得到的嵌入。
    # entrez_p_file = '../data/AD_GWAS_data/bedtools_data/entrez_p_sentence.tsv'
    # entrez_to_p, entrez_to_tag_set = read_entrez_p(entrez_p_file)
    #
    # entrez_bag_embedding_file = '../result/entrez.MPA-CPA.total.embedding.txt'
    # entrez_list, trend_embedding_array, entrez_to_trend_embedding = read_embedding(entrez_bag_embedding_file, None, False)
    # print(f'trend_embedding: {trend_embedding_array.shape}')


    # 3.
    # entrez_to_merge_embedding = trend_bag_embedding_merge(entrez_to_trend_embedding,
    #                                                       entrez_to_bag_embedding,
    #                                                       0.5)
    #
    # # 4.
    # embedding_save_file = f'../result/embedding/trend_bag_embedding.txt'
    # save_embedding(entrez_to_merge_embedding, embedding_save_file)
    #
    # # 5.
    # plot_save_file = f'../result/trend_embedding_plot/trend_bag_embedding.png'
    # save_plot(plot_save_file, embedding_save_file, entrez_to_p)


if __name__ == '__main__':
    main()

