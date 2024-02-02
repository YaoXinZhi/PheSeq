# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 22/09/2021 19:44
@Author: XINZHI YAO
"""

import os
import time
import argparse

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


# read embedding file
def read_embedding(embedding_file: str):
    print('Reading embedding file.')
    embedding_list = [ ]
    entrez_list = [ ]
    term_embedding_dict = {}
    with open(embedding_file) as f:
        for line in f:
            l = line.strip().split('\t')
            if len(l) != 2:
                print(l)
                input()
                continue

            term = '-'.join(l[ 0 ].split('-')[ 1: ])
            entrez_list.append(l[ 0 ])

            embedding = list(map(float, l[ 1 ].split()))
            embedding_list.append(embedding)
            term_embedding_dict[ term ] = embedding

    print(f'data size: {len(entrez_list):,}, feature size: {len(embedding_list[ 0 ])}')
    return entrez_list, np.array(embedding_list), term_embedding_dict


# read entrez_p_file
def read_entrez_p(entrez_to_p_file: str):
    print('Reading Entrez p-value file.')
    entrez_p_dict = {}
    entrez_to_symbol = {}
    entrez_bag = defaultdict(list)
    entrez_tag = defaultdict(list)

    entrez_tag_set = defaultdict(set)
    with open(entrez_to_p_file) as f:
        for line in f:
            l = line.strip().split('\t')
            if line.startswith('GENE_LINE'):
                symbol, entrez, p = l[ 1 ], l[ 2 ], float(l[ -1 ])
                entrez_p_dict[ entrez ] = p
                entrez_to_symbol[ entrez ] = symbol
            else:
                pmid = l[ 0 ]
                sentence = l[ 1 ]
                tags = eval(l[ 2 ])

                # print(tags)
                entrez_tag_set[ entrez ].update(tags)

                entrez_bag[ entrez ].append((pmid, sentence))
                entrez_tag[ entrez ].append(tags)
    print(f'data size: {len(entrez_p_dict)}, \
            min_p: {min(entrez_p_dict.values())}, \
            max_p: {max(entrez_p_dict.values())}.')
    return entrez_p_dict, entrez_tag_set

def get_idx_soft_factor(entrez_count: int):
    mid_idx = int(entrez_count/2)
    factor_list = [idx - mid_idx for idx in range(entrez_count)]
    return factor_list

def soft_p_vec(true_p: float):
    if true_p < 0.01:
        factor = -20
    elif 0.01 <= true_p < 0.02:
        factor = -18
    elif 0.02 <= true_p < 0.04:
        factor = -16
    elif 0.04 <= true_p < 0.05:
        factor = -15
    elif 0.05 <= true_p < 0.06:
        factor = -13
    elif 0.06 <= true_p < 0.08:
        factor = -10
    elif 0.08 <= true_p < 0.10:
        factor = -8
    elif 0.10 <= true_p < 0.14:
        factor = -5
    elif 0.14 <= true_p < 0.18:
        factor = 0
    elif 0.18 <= true_p < 0.20:
        factor = 5
    elif 0.20 <= true_p < 0.40:
        factor = 8
    elif 0.40 <= true_p < 0.60:
        factor = 10
    elif 0.60 <= true_p < 0.80:
        factor = 15
    elif 0.80 <= true_p <= 1:
        factor = 20
    else:
        factor = 0

# average_embedding
def get_avg_embedding(entrez_to_term: dict, term_embedding_dict: dict,
                      entrez_p_dict: dict, use_tag_type: set,
                      _beta=0.7, _mu_factor=5,
                      _sigma=1, only_base=False,
                      normal_bool=True,
                      _use_log_P=False,
                      _distinctive_normal=False,
                      _distinctive_threshold=0.1,
                      _distinctive_factor=200,
                      soft_factor_normal=False,):
                      # soft_up=500,
                      # soft_down=-500):
    embedding_size = len(term_embedding_dict[ list(term_embedding_dict.keys())[ 0 ] ])
    # print(embedding_size)
    # input()
    if _distinctive_normal:
        print('Use distinctive normal.')
        print(f'Distinctive threshold: {_distinctive_threshold}')

    if soft_factor_normal:
        print('soft_factor_normal')
        entrez_count = len(entrez_p_dict.keys())
        # soft_factor_list = np.linspace(soft_down, soft_up, entrez_count)
        soft_idx_factor_list = get_idx_soft_factor(entrez_count=entrez_count)

    else:
        soft_idx_factor_list = []

    sorted_entrez = sorted(entrez_p_dict.keys(), key=lambda x: entrez_p_dict[x])

    print('Most simple P embedding.')

    avg_count = 0
    entrez_to_avg_embedding = {}
    _entrez_to_single_p_embedding = {}
    _entrez_to_single_term_embedding = {}
    for idx, entrez in enumerate(sorted_entrez):

        # 通过p值产生正态分布初始向量
        term_set = entrez_to_term[entrez]
        # P: 0-1
        p = entrez_p_dict[ entrez ]
        # mu:
        if p == 0:
            p = 1e-350

        # if normal_bool:
        #     if _use_log_P:
        #         mu = -np.log(p) * _mu_factor
        #     else:
        #         if _distinctive_normal:
        #             if p < _distinctive_threshold:
        #                 mu = p * _distinctive_factor
        #             else:
        #                 mu = p * - _distinctive_factor
        #         elif soft_factor_normal:
        #             mu = p * soft_idx_factor_list[idx]
        #         else:
        #             mu = p * _mu_factor

            # mu = -np.log(p) * _mu_factor
        #     normal_embedding = np.random.normal(mu, _sigma, embedding_size)
        # else:
        #     normal_embedding = np.array([p]*embedding_size)

        # most simple way to generate the p vector

        mu = p * _mu_factor
        normal_embedding = np.random.normal(mu, _sigma, embedding_size)

        # save single p embedding
        _entrez_to_single_p_embedding[entrez] = normal_embedding

        sum_embedding = np.zeros(embedding_size)
        appear_term_count = 0
        for term in term_set:

            term_phrase = term[ 0 ]
            term_type = term[ 1 ]
            if term_embedding_dict.get(term_phrase) and term_type in use_tag_type:
                term_embedding = term_embedding_dict[ term_phrase ]
                sum_embedding += term_embedding
                appear_term_count += 1

        # save single MPA/CPA/HPO embedding
        if appear_term_count == 0:
            # _entrez_to_single_term_embedding[entrez] = np.array([0]*embedding_size)
            # fixme: 10-26 add empty embedding
            _entrez_to_single_term_embedding[entrez] = np.random.normal(0, 1, embedding_size)
        else:
            _entrez_to_single_term_embedding[entrez] = sum_embedding / appear_term_count

        # if idx %10 == 0:
        #     print(entrez)
        #     print(_beta)
        #     print(sum_embedding[:5])
        #     print(normal_embedding[:5])
        if appear_term_count != 0:
            if not only_base:
                # 10-26 test is right.
                # print(sum_embedding[:10])
                # print(appear_term_count)
                avg_embedding = sum_embedding / appear_term_count
                # print(avg_embedding[:10])
                # right
                # print(normal_embedding[:10])
                avg_embedding = _beta * normal_embedding + (1 - _beta) * avg_embedding
                # print(avg_embedding[:10])
                # input()
            else:
                avg_embedding = normal_embedding
            avg_count += 1
        else:
            # fixme 10-26 add empty embedding
            avg_embedding = np.random.normal(0, 1, embedding_size)
            avg_embedding = _beta * normal_embedding + (1 - _beta) * avg_embedding

        # if idx %10 == 0:
        #     print(appear_term_count)
        #     print(avg_embedding[:5])
        #     input()

        entrez_to_avg_embedding[ entrez ] = avg_embedding
    print(f'Average count: {avg_count}, Random Count: {len(entrez_to_term.keys()) - avg_count}.')
    return entrez_to_avg_embedding, _entrez_to_single_p_embedding, _entrez_to_single_term_embedding


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

    parser = argparse.ArgumentParser(description='Tendency bag embedding merge.')
    parser.add_argument('--summary_data_path', dest='summary_data_path', type=str,
                        default='../data/AD_GWAS_data/other_GWAS/CNCR_data/NG_2019_bedtools',
                        help='default: ../data/AD_GWAS_data/other_GWAS/CNCR_data/NG_2019_bedtools')
    parser.add_argument('--base_embedding_save_path', dest='base_embedding_save_path',
                        default='../result/NG_2019_result',
                        help='default: ../result/NG_2019_result')
    parser.add_argument('--embedding_save_dir', dest='embedding_save_dir',
                        default='p_norm_hpo_embedding_distinctive',
                        help='default: p_norm_hpo_embedding_distinctive')
    # parser.add_argument('--distinctive_factor', dest='distinctive_factor',
    #                     type=int, default=20,
    #                     help='default: 20')

    parser.add_argument('--save_plot', dest='save_plot', action='store_true',
                        default=False,
                        help='save_plot, default: False')
    parser.add_argument('--plot_save_dir', dest='plot_save_dir',
                        default='distinctive_plot',
                        help='default: distinctive_plot')

    args = parser.parse_args()

    # distinctive factor
    distinctive_normal = False
    distinctive_threshold = 0.1
    distinctive_factor = 20

    # soft factor normal
    soft_factor_normal = False
    soft_up = 500
    soft_down = -500

    # beta 寻优
    ONLY_BASE = False
    normal_embedding_bool = True

    use_log_P = False

    # do not merge normal embedding and term embedding
    only_save_single_entrez_embedding = False

    # change 1
    # summary_data_path = '../data/AD_GWAS_data/other_GWAS/CNCR_data/NG_2019_bedtools'
    # summary_data_path = '../data/AD_GWAS_data/other_GWAS/CNCR_data/Genet_2021_bedtools'

    # change 2
    # base_embedding_save_path = '../result/NG_2019_result'
    # embedding_save_path = '../result/Genet_2021_result'

    if not os.path.exists(args.base_embedding_save_path):
        os.mkdir(args.base_embedding_save_path)

    # change 3
    # merge_embedding_save_path = f'{embedding_save_path}/p_norm_hpo_embedding'
    # merge_embedding_save_path = f'{args.base_embedding_save_path}/p_norm_hpo_embedding_distinctive'
    merge_embedding_save_path = f'{args.base_embedding_save_path}/{args.embedding_save_dir}'

    if not os.path.exists(merge_embedding_save_path):
        os.mkdir(merge_embedding_save_path)

    # 1. 读取entrez 到 MPA，CPA文件
    # save_tag_type = {'MPA', 'CPA', 'Disease'}
    save_tag_type = {'MPA', 'CPA', 'Disease', 'Phenotype'}
    # 10-13 add HPO
    entrez_p_file = f'{args.summary_data_path}/entrez_p_sentence.HPO.tsv'

    # change
    entrez_to_p, entrez_to_tag_set = read_entrez_p(entrez_p_file)

    # 2. 读取 CPA，MPA的嵌入
    MPA_CPA_HPO_embedding_file = f'{args.base_embedding_save_path}/MPA.CPA.DIS.HPO.embedding.txt'
    MCD_list, _, term_to_embedding = read_embedding(MPA_CPA_HPO_embedding_file)

    # 3. 根据每个entrez对应的所有MPA，CPA计算平均嵌入
    # Average count: 4021, Random Count: 580.

    # 当ONLY_BASE=True，只保留基本嵌入，即通过高斯分布随机生成嵌入的结果，则beta参数无效
    # ONLY_BASE = True

    # 3.5 批量保存基本嵌入 看看mu和sigma的设置哪个比较合理
    # ONLY_BASE = True
    # for mu_factor in range(1, 50, 5):
    #     if mu_factor == 1:
    #         continue
    #     # for sigma in range(10):
    #     #     sigma = sigma * 2 + 0.1
    #     #     if sigma > 3:
    #     #         break
    #     for sigma in {0.1}:
    #
    #         print(f'{mu_factor}-{sigma}')
    #         entrez_to_MC_embedding = get_avg_embedding(entrez_to_tag_set, term_to_embedding,
    #                                                    entrez_to_p, save_tag_type,
    #                                                    _beta=0.4, _mu_factor=mu_factor,
    #                                                    _sigma=sigma, only_base=ONLY_BASE,)
    #
    #         # 4. 保存嵌入
    #         embedding_save_file = f'../result/only_normal_embedding/entrez.base-embedding.{mu_factor}-{sigma:.2f}.txt'
    #         save_embedding(entrez_to_MC_embedding, embedding_save_file)

            # # 保存可视化图片
            # plot_save_file = f'../result/only_normal_embedding_plot/base.{mu_factor}-{sigma:.2f}.png'
            # save_plot(plot_save_file, embedding_save_file, entrez_to_p)


    for beta in range(0, 100, 1):
        beta = beta * 0.1 + 0.00
        if beta > 1:
            break

        entrez_to_MC_embedding, \
        entrez_to_single_p_embedding, \
        entrez_to_single_term_embedding = get_avg_embedding(entrez_to_tag_set, term_to_embedding,
                                              entrez_to_p, save_tag_type, _beta=beta,
                                              _mu_factor=5,
                                              _sigma=0.1, only_base=ONLY_BASE,
                                              normal_bool=normal_embedding_bool,
                                              # _use_log_P=use_log_P,
                                              # _distinctive_normal=distinctive_normal,
                                              # _distinctive_threshold=distinctive_threshold,
                                              # _distinctive_factor=distinctive_factor,
                                              # soft_factor_normal=soft_factor_normal,
                                              # soft_up=soft_up,
                                              # soft_down=soft_down
                                                            )

        # 4. 保存嵌入
        # embedding_save_file = f'../result/log_p_vec_embedding/entrez.MPA-CPA.total.{beta:.2f}.embedding.txt'
        embedding_save_file = f'{merge_embedding_save_path}/entrez.MPA-CPA.total.{beta:.2f}.embedding.txt'
        # embedding_save_file = '../result/entrez.base-embedding.txt'
        save_embedding(entrez_to_MC_embedding, embedding_save_file)

        if only_save_single_entrez_embedding:
            print('only_save_single_entrez_embedding')
            single_p_save_file = '../data/DME_embedding/entrez.p-value.embedding.txt'
            single_term_save_file = '../data/DME_embedding/entrez.term.embedding.txt'

            save_embedding(entrez_to_single_p_embedding, single_p_save_file)
            save_embedding(entrez_to_single_term_embedding, single_term_save_file)

            exit()

        # 保存可视化图片
        if args.save_plot:
            plot_save_file = f'{args.base_embedding_save_path}/{args.plot_save_dir}/MPA-CAP.total.{beta:.2f}.png'
            save_plot_new(plot_save_file, embedding_save_file, entrez_to_p)

    # ONLY_BASE = True
    # normal_embedding = False
    # entrez_to_MC_embedding = get_avg_embedding(entrez_to_tag_set, term_to_embedding,
    #                                            entrez_to_p, save_tag_type,
    #                                            _beta=0.1,
    #                                            _mu_factor=5,
    #                                            _sigma=0.1, only_base=ONLY_BASE)
    #
    # # 4. 保存嵌入
    # embedding_save_file = '../result/entrez.MPA-CPA.total.embedding.beta-0.3.txt'
    # # embedding_save_file = '../result/entrez.MPA-CPA.total.logp.embedding.txt'
    # # embedding_save_file = '../result/entrez.base-embedding.txt'
    # save_embedding(entrez_to_MC_embedding, embedding_save_file)

    # 保存可视化图片
    # plot_save_file = f'../result/base-embedding_plot/MPA-CPA.total.beta-0.3.png'
    # # plot_save_file = f'../result/base-embedding_plot/MPA-CPA.logp.total.png'
    # # plot_save_file = f'../result/base-embedding_plot/base-embedding.png'
    # # save_plot(plot_save_file, embedding_save_file, entrez_to_p)
    # save_plot_new(plot_save_file, embedding_save_file, entrez_to_p)

if __name__ == '__main__':
    main()
