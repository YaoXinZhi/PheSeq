# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 08/02/2023 11:34
@Author: XINZHI YAO
"""

import os
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from itertools import cycle

def read_cluster_file(cluster_file):

    entrez_to_cluster = {}
    with open(cluster_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split('\t')
            entrez = l[0]
            cluster_name = l[1]

            entrez_to_cluster[entrez] = cluster_name
    return entrez_to_cluster

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


def main(cluster_file: str, embedding_file: str, save_path: str):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    cluster_save_path = f'{save_path}/Embedding-cluster.png'

    entrez_to_cluster = read_cluster_file(cluster_file)

    entrez_list, embedding_list = read_embedding(embedding_file)

    embedding_matrix = np.array([embedding for embedding in embedding_list])
    cluster_name_list = [entrez_to_cluster[entrez] for entrez in entrez_list]

    start_time = time.time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    embedding_tsne = tsne.fit_transform(embedding_matrix)
    end_time = time.time()
    print(f'TSNE done, cost: {end_time - start_time:.2f} s.')

    cycol = cycle('bgrcmk')
    colors = [ random_color() for _ in range(len(cluster_name_list)) ]
    idx_to_color = {idx: colors[ i ] for i, idx in enumerate(set(cluster_name_list))}

    idx_list = [ ]
    color_list = [ ]
    for idx, color in idx_to_color.items():
        idx_list.append(idx)
        color_list.append(color)

    # 开始画图
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)

    type_list = [ ]
    for idx, color in idx_to_color.items():
        idx_embedding_matrix = np.array(
            [ embedding for i, embedding in enumerate(embedding_tsne) if cluster_name_list[ i ] == idx ])
        # plt.scatter(idx_embedding_matrix[:,0],idx_embedding_matrix[:,1], color=idx_to_color[idx])
        ax.scatter(idx_embedding_matrix[ :, 0 ], idx_embedding_matrix[ :, 1 ], color=idx_to_color[ idx ])
        type_list.append(f'{idx}-{idx_to_type[ idx ]}')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([ box.x0, box.y0, box.width * 0.8, box.height ])

    # Put a legend to the right of the current axis
    ax.legend(type_list, loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.legend(type_list, loc='center left',  bbox_to_anchor=(1,1))
    plt.show()
    plt.savefig(cluster_save_path)
    print(f'Embedding cluster saved: {cluster_save_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-cf', dest='cluster_file', required=True,
                        help='tab split file, include embedding_name, and cluster_name.')

    parser.add_argument('-ef', dest='embedding_file', required=True,
                        help='embedding file, tab split embedding_name and embedding_vec.')

    parser.add_argument('-sp', dest='save_path', required=True,
                        help='path used to save plot.')

    args = parser.parse_args()

    main(args.embedding_file, args.p_value_file, args.save_path)

