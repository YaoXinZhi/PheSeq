# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 31/01/2024 11:39
@Author: yao
"""

# import os
import time
import argparse
import networkx as nx
from node2vec import Node2Vec

def read_edge_file(file_path, delimiter='\t'):
    G = nx.Graph()

    with open(file_path) as f:
        for line in f:
            l = line.strip().split(delimiter)
            node1, node2 = l[0], l[1]

            G.add_edge(node1, node2)

    print(f'{len(G.nodes):,} nodes, {len(G.edges):,} edges.')
    return G

def calculate_node2vec_embedding(graph, embedding_size):
    print(f'calculating embeddings.')
    # 使用node2vec算法
    node2vec = Node2Vec(graph, dimensions=embedding_size, walk_length=10, num_walks=200, workers=6)

    # 拟合模型（Fit model）
    model = node2vec.fit(window=5, min_count=1, batch_words=6)

    # 获取所有节点的嵌入
    embeddings = {node: model.wv[str(node)] for node in graph.nodes()}

    return embeddings


# 保存嵌入到文件
def save_embeddings(embeddings, output_file):
    with open(output_file, 'w') as f:
        for node, embed_vector in embeddings.items():
            embed_str = ' '.join(map(str, embed_vector))
            f.write(f"{node} {embed_str}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge_file', required=True)
    parser.add_argument('--save_file', required=True)
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='default: 128')
    args = parser.parse_args()

    print(args)
    start_time = time.time()

    graph = read_edge_file(args.edge_file, delimiter='\t')

    embeddings = calculate_node2vec_embedding(graph, args.embedding_size)

    save_embeddings(embeddings, args.save_file)

    end_time = time.time()
    print(f'time cost: {end_time-start_time:.2f}s.')
    print(f'{args.save_file} saved.')


if __name__ == '__main__':
    main()

