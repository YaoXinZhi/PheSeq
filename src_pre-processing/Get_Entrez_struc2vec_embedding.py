# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 12/10/2021 20:53
@Author: XINZHI YAO
"""

import os

import numpy as np

def read_gene2ensemble(gene2ensemble_file: str):

    print('Reading gne2ensemble file.')
    ensemble_to_entrez = {}
    with open(gene2ensemble_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split('\t')
            tax_id = l[0]

            if tax_id != '9606':
                continue
            entrez_id = l[1]
            protein_ensemble_id = l[-1].split('.')[0]

            ensemble_to_entrez[protein_ensemble_id] = entrez_id

    return ensemble_to_entrez

def read_embedding(embedding_file: str):
    print('Reading embedding file.')
    embedding_list = [ ]
    entrez_list = [ ]
    entrez_to_embedding = {}
    with open(embedding_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split()

            entrez = l[0]
            entrez_list.append(entrez)

            embedding = l[1:]
            embedding = list(map(float, embedding))

            embedding_list.append(embedding)
            entrez_to_embedding[entrez] = embedding
    print(f'data size: {len(entrez_list):,}, feature size: {len(embedding_list[ 0 ])}')
    return entrez_list, np.array(embedding_list), entrez_to_embedding

def read_node_file(node_file: str):

    print('reading node file.')
    idx_to_ensemble = {}
    with open(node_file) as f:
        f.readline()
        for line in f:
            idx, ensemble = line.strip().split('\t')

            if not ensemble.startswith('9606'):
                continue
            ensemble = ensemble.split('.')[1]

            idx_to_ensemble[idx] = ensemble
    return idx_to_ensemble


def get_entrez_to_embedding(idx_to_ensemble: dict, idx_to_embedding: dict,
                            ensemble_to_entrez: dict, save_file: str):

    print('getting entrez embedding.')
    match_count = 0

    # print(ensemble_to_entrez)
    # input()
    with open(save_file, 'w') as wf:

        for idx, embedding in idx_to_embedding.items():

            # print(idx)
            ensemble = idx_to_ensemble[idx]
            # print(ensemble)
            # input()
            if ensemble_to_entrez.get(ensemble):
                entrez = ensemble_to_entrez[ensemble]
                match_count += 1

                embedding_wf = ' '.join(map(str, embedding))
                wf.write(f'{entrez}\t{embedding_wf}\n')
    print(f'{save_file} save done. {match_count:,}/{len(idx_to_embedding):,} embedding matched.')

def main():

    gene2ensemble_file = '../data/gene2ensembl'

    node_file = '../data/graph_embedding/STRING_PPI/node_list.txt'

    embedding_file = '../data/graph_embedding/STRING_PPI/STRING_PPI_struc2vec_number_walks64_walk_length16_dim100.txt'

    save_file = '../data/graph_embedding/STRING_PPI/entrez_embedding.struc2vec.txt'

    ensemble_to_entrez = read_gene2ensemble(gene2ensemble_file)

    _, _, idx_to_embedding = read_embedding(embedding_file)

    idx_to_ensemble = read_node_file(node_file)

    # ../data/graph_embedding/STRING_PPI/entrez_embedding.struc2vec.txt save done.
    # 13,234/15,131 embedding matched.

    # AD related genes: 4601
    # 3855 genes can find embedding in here.
    get_entrez_to_embedding(idx_to_ensemble, idx_to_embedding, ensemble_to_entrez,
                            save_file)

if __name__ == '__main__':
    main()
