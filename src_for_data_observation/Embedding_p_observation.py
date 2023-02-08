# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 08/02/2023 11:12
@Author: XINZHI YAO
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


# read embedding file
def read_embedding(embedding_file: str):
    embedding_list = [ ]
    entrez_list = [ ]
    with open(embedding_file) as f:
        for line in f:
            l = line.strip().split('\t')
            if len(l) != 2:
                print(line)
                raise ValueError(f'Wrong line format.')
            entrez_list.append(l[ 0 ])

            embedding = list(map(float, l[ 1 ].split()))
            embedding_list.append(embedding)
    print(f'data size: {len(entrez_list):,}, feature size: {len(embedding_list[ 0 ])}')
    return entrez_list, np.array(embedding_list)


# read entrez_p_file
def read_entrez_p(entrez_p_file: str):
    entrez_to_p = {}
    with open(entrez_p_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split('\t')
            entrez = l[ 0 ]
            p = float(l[ 1 ])

            entrez_to_p[ entrez ] = p
    return entrez_to_p


def main(embedding_file: str, p_value_file: str, save_path: str):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    embedding_p_association_plot = f'{save_path}/embedding-p.association.png'

    entrez_list, embedding_list = read_embedding(embedding_file)
    entrez_to_p = read_entrez_p(p_value_file)

    print('TSNE running')
    start_time = time.time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    embedding_tsne = tsne.fit_transform(embedding_list)
    end_time = time.time()
    print(f'TSNE done, cost: {end_time - start_time:.2f} s.')

    # p-value process
    p_list = [ entrez_to_p[ entrez ] for entrez in entrez_list ]

    # split data based on p-value
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

    # 可视化
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    images = list()
    images.append([
        axes.scatter(sig_embedding[ :, 0 ], sig_embedding[ :, 1 ], marker='o', c=sig_p_list, cmap='Oranges_r',
                     alpha=0.3),
        axes.scatter(insig_embedding[ :, 0 ], insig_embedding[ :, 1 ], marker='o', c=insig_p_list, cmap='summer',
                     alpha=0.7),
    ])

    fig.colorbar(images[ 0 ][ 1 ], ax=axes, fraction=.05, pad=0.15)
    fig.colorbar(images[ 0 ][ 0 ], ax=axes, fraction=.05, pad=0.15)

    plt.savefig(embedding_p_association_plot)
    print(f'Embedding-p association observation was saved: "{embedding_p_association_plot}".')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ef', dest='embedding_file', required=True,
                        help='embedding file, tab split embedding_name and embedding_vec.')

    parser.add_argument('-pf', dest='p_value_file', required=True,
                        help='p-value file, tab split embedding_name and p-value.')

    parser.add_argument('-sp', dest='save_path', required=True,
                        help='path used to save plot.')

    args = parser.parse_args()

    main(args.embedding_file, args.p_value_file, args.save_path)
