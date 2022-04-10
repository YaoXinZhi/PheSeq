# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 11/10/2021 16:17
@Author: XINZHI YAO
"""


import re

import os
import shutil
import argparse

from collections import defaultdict

def read_diseases_file(diseases_file: str):

    entrez_to_source = defaultdict(set)
    with open(diseases_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split('\t')

            symbol, entrez, source = l
            entrez_to_source[entrez].add(source)
    return entrez_to_source



def copy_cls(save_para_set: set, source_path: str, target_path: str):

    if os.path.exists(target_path):
        shutil.rmtree(target_path)
        print(f'{target_path} created.')

    if not os.path.exists(target_path):
        os.mkdir(target_path)
        print(f'{target_path} removed.')

    for (trend_beta, base_beta) in save_para_set:
        base_path = f'TrendBeta-{trend_beta:.2f}_beta-{base_beta:.2f}_p-vec'
        cls_path = f'{source_path}/{base_path}'

        if not os.path.exists(cls_path):
            base_path = f'TrendBeta-{trend_beta:.1f}_beta-{base_beta:.1f}_p-vec'
            cls_path = f'{source_path}/{base_path}'

        save_path = f'{target_path}/{base_path}'

        shutil.copytree(cls_path, save_path)
        print(f'{base_path} copy done.')

class pred_cls:
    def __init__(self):
        self.f = None
        self.t = None
        self.p_ture = None
        self.p_pred = None

def read_entrez_p(entrez_p_file: str):
    print(f'Reading entrez_p_file: {entrez_p_file}')

    entrez_to_p = {}
    entrez_to_symbol = {}
    entrez_bag = defaultdict(list)
    entrez_tag = defaultdict(list)
    with open(entrez_p_file) as f:
        for line in f:
            l = line.strip().split('\t')
            if line.startswith('GENE_LINE'):
                symbol, entrez, p = l[ 1 ], l[ 2 ], float(l[ -1 ])

                if p == 0:
                    p = 1e-38

                entrez_to_p[ entrez ] = p
                entrez_to_symbol[ entrez ] = symbol
            else:
                pmid = l[ 0 ]
                sentence = l[ 1 ]
                tags = eval(l[ 2 ])

                entrez_bag[ entrez ].append((pmid, sentence))
                entrez_tag[ entrez ].append(tags)
    return entrez_to_p, entrez_bag, entrez_tag, entrez_to_symbol

def read_pred_file(pred_file: str):

    entrez_to_pred = defaultdict(pred_cls)
    with open(pred_file) as f:
        f.readline()
        for line in f:
            try:
                entrez, f, t, p_ture, p_pred = line.strip().split('\t')
            except:
                print(line)
            f = float(f)
            t = int(float(t))
            p_ture = float(p_ture)
            p_pred = float(p_pred)

            entrez_to_pred[entrez].f = f
            entrez_to_pred[entrez].t = t
            entrez_to_pred[entrez].p_ture = p_ture
            entrez_to_pred[entrez].p_pred = p_pred
    return entrez_to_pred

def bag_to_write(bag):
    wf_line = [f'{pmid}-{sentence}' for pmid, sentence in bag]
    sentence_count = len(wf_line)
    if not wf_line:
        wf_line = ['None']
    return '  '.join(wf_line), sentence_count

def tag_to_write(tag_set_list: set):

    tag_wf = []
    for tag_set in tag_set_list:
        for tag in tag_set:
            tag_type = tag[1]
            # fixme: change tag write
            if tag_type in {'MPA', 'CPA', 'Phenotype'}:
                mention = tag[0]
                if tag_type in {'MPA', 'CPA'}:
                    tag_wf.append(f'{tag_type}|{mention}')
                elif tag_type in {'Phenotype'}:
                    tag_id = tag[2]
                    tag_wf.append(f'{tag_type}|{tag_id}|{mention}')

            if len(tag) == 5:
                if tag[3].startswith('GO:'):
                    mention = tag[0]
                    proffered = tag[1]
                    tag_id = tag[3]
                    tag_wf.append(f'{tag_id}|{proffered}|{mention}')

    tag_count = len(set(tag_wf))
    if not tag_wf:
        tag_wf = ['None']
    else:
        tag_wf = set(tag_wf)
    # double space
    return '  '.join(tag_wf), tag_count


def is_available(path):

    if path.endswith('F-P_scatter_dir'):
        return False

    file_list = os.listdir(path)
    for _file in file_list:
        if _file.endswith('png'):
            return True
    return False


def ensemble_statistic(selected_path: str, save_path: str, save_prefix: str,
                       entrez_bag: dict, entrez_tag: dict, entrez_to_symbol: dict,
                       diseases_source: dict):
    print('vote statistic.')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    statistic_file = f'{save_path}/{save_prefix}.ensemble.tsv'

    readme_file = f'{save_path}/{save_prefix}.README.txt'
    wf_readme = open(readme_file, 'w')
    wf_readme.write(f'Selected parameters.\n')

    entrez_to_statistic = defaultdict(list)
    entrez_to_vote = defaultdict(int)
    para_list = []
    path_list = os.listdir(selected_path)

    for path in path_list:

        if path.endswith('F-P_scatter_dir'):
            continue

        # print(f'Processing: {path}')
        # if not is_available(f'{selected_path}/{path}'):
        #     continue

        trend_beta, base_beta = re.findall(r'\d\.\d\d', path)
        wf_readme.write(f'Trend_beta-{trend_beta}\tBase_beta-{base_beta}\n')
        para_list.append(f'Trend_beta-{trend_beta}-Base_beta-{base_beta}')

        # pred_file = f'{selected_path}/{path}/TrendBeta-{trend_beta}.BaseBeta-{base_beta}.p-pred.tsv'
        pred_file = f'{selected_path}/{path}/TrendBeta-{trend_beta}.BaseBeta-{base_beta}.pred.tsv'
        if not os.path.exists(pred_file):
            pred_file = f'{selected_path}/{path}/TrendBeta-{trend_beta}.BaseBeta-{base_beta}.pred.txt'

        entrez_to_pred = read_pred_file(pred_file)

        for entrez, pred in entrez_to_pred.items():
            entrez_to_statistic[entrez].append((pred.t, pred.f, pred.p_pred, pred.p_ture))
            if pred.t == 1:
                entrez_to_vote[entrez] += 1

    wf_readme.close()

    # print(diseases_source)
    # 根据投票总数排序
    entrez_sort = sorted(entrez_to_vote.keys(), key=lambda x: entrez_to_vote[x], reverse=True)
    # 根据MPA CPA注释数量排序
    with open(statistic_file, 'w') as wf:
        # head line
        para_list.insert(0, 'Symbol')
        para_list.insert(0, 'Entrez')

        para_list.append('Total Vote\t'
                         'Most significant P-value\tTrue P-value\t'
                         'Sentence Count\tSentence\t'
                         'Tag Count\tMPA/CPA tag\tDISEASES Score')
        head_line = '\t'.join(para_list)
        wf.write(f'{head_line}\n')

        for entrez in entrez_sort:

            p_pred_list = []

            f_list = []
            t_list = []

            symbol = entrez_to_symbol[entrez]

            write_list = [entrez, symbol]

            for pred in entrez_to_statistic[entrez]:
                t, f, p_pred, p_true = pred
                f_list.append(f)
                t_list.append(t)

                p_pred_list.append(p_pred)
                # t for each classifier
                write_list.append(t)

            # total vote count
            write_list.append(entrez_to_vote[entrez])
            # most significant p-value
            write_list.append(min(p_pred_list))
            # true p-value
            write_list.append(p_true)

            # sentence bag
            bag = entrez_bag[entrez]
            bag_wf, sentence_count = bag_to_write(bag)
            write_list.append(sentence_count)
            write_list.append(bag_wf)

            # MPA_CPA_tag
            tag_list = entrez_tag[entrez]
            tag_wf, tag_count = tag_to_write(tag_list)
            write_list.append(tag_count)
            write_list.append(tag_wf)

            # ADD DISEASES Score
            diseases_score = 0
            if diseases_source.get(entrez):
                # print(entrez)
                # print(diseases_source[entrez])
                # input()
                source_set = diseases_source[entrez]
                # Knowledge 2 score
                # text mining 1 score
                # experiment 1.5 score
                if 'Knowledge' in source_set:
                    diseases_score += 2
                if 'Textmining' in source_set:
                    diseases_score += 1
                if 'Experiments' in source_set:
                    diseases_score += 1.5

            write_list.append(diseases_score)

            wf_line = '\t'.join(map(str, write_list))

            wf.write(f'{wf_line}\n')

    print(f'{statistic_file} save done.')


def main():

    parser = argparse.ArgumentParser(description='Model Ensemble.')


    parser.add_argument('--log_path', dest='log_path',
                        default='../log/selected_best_alpha-log',
                        help='../log/selected_best_alpha-log')

    parser.add_argument('--summary_path', dest='summary_path',
                        default='../data/AD_GWAS_data/other_GWAS/CNCR_data/NG_2019_bedtools',
                        help='../data/AD_GWAS_data/other_GWAS/CNCR_data/NG_2019_bedtools')

    parser.add_argument('--output_path', dest='output_path', type=str,
                        default='../result/ensemble_result/test_result',
                        help='default ../result/ensemble_result/test_result')
    parser.add_argument('--output_prefix', dest='output_prefix',
                        default=None,
                        help='default: None')

    # 根据DISEASES数据库的结果进行打分
    parser.add_argument('--diseases_source', dest='diseases_source', type=str,
                        default='../data/Supplementary_experiment/BreastCancer_dir/BreastCancer.DIS.txt',
                        help='../data/Supplementary_experiment/BreastCancer_dir/BreastCancer.DIS.txt, could be "None".')

    args = parser.parse_args()

    if args.output_prefix is None:
        output_prefix = args.output_path.split('/')[-1]
    else:
        output_prefix = args.output_prefix

    # 1. read DISEASES Source
    if args.diseases_source != 'None':
        entrez_to_source = read_diseases_file(args.diseases_source)
    else:
        entrez_to_source = {}

    # 2. read entrez_to_sentence
    entrez_p_file = f'{args.summary_path}/entrez_p_sentence.HPO.OGER.tsv'
    entrez_to_p, entrez_bag, entrez_tag, entrez_to_symbol = read_entrez_p(entrez_p_file)

    ensemble_statistic(args.log_path, args.output_path, output_prefix,
                       entrez_bag, entrez_tag, entrez_to_symbol, entrez_to_source)


if __name__ == '__main__':
    main()
