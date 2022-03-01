# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 19/10/2021 21:13
@Author: XINZHI YAO
"""

"""
该代码用于转换GWAS Summary 数据为以下格式
GENE_LINE:  Symbol  entrez  p
PMID1   sent1
PMID2   sent2
"""

from collections import defaultdict

import gzip


"""
1. GWAS data to position data for bedtools

chr1 start_offset start_offset
"""

# GWAS_file = 'AD_sumstats_Jansenetal_2019sept.txt'
# chr_col = 1
# start_col = 2
# p_line = 7

GWAS_file = 'PGCALZ2sumstatsExcluding23andMe.txt'
chr_col = 0
start_col = 1
p_line = 5

save_file = 'gene_pos.bed'

print('reading original GWAS summary data.')
pos_to_most_sig_p = {}
with open(GWAS_file) as f, open(save_file, 'w') as wf:
        f.readline()
        for line in f:
            l = line.strip().split('\t')
            chr_num = l[chr_col]
            chr_num_wf = f'chr{l[chr_col]}'
            offset = l[start_col ]

            p = float(l[p_line])

            if pos_to_most_sig_p.get((chr_num, offset)):
                if p < pos_to_most_sig_p[(chr_num, offset)]:
                    pos_to_most_sig_p[(chr_num, offset)] = p
            else:
                pos_to_most_sig_p[(chr_num, offset)] = p

            wf.write(f'{chr_num_wf}\t{offset}\t{offset}\n')


"""
2. 用bedtoos将染色体号和位置对应到基因symbol
bedtools intersect -a gene_pos.bed  -b ../../../hg38_data/protein_coding.hg38.position  -wa -wb| bedtools groupby -i - -g 1-3 -c 7 -o collapse > gene_pos.symbol.tsv
"""


"""
3. 读取结果
"""

mapped_gene_file = 'gene_pos.symbol.tsv'

chr_col_result = 0
pos_col_result = 1
symbol_col = -1

symbol_set = set()
pos_to_symbol = defaultdict(set)
with open(mapped_gene_file) as f:
    for line in f:
        l = line.strip().split('\t')

        chr_num = l[chr_col_result][3:]
        pos = l[pos_col_result]

        # multi-gene in same position.
        gene_symbol = l[symbol_col].split(',')

        pos_to_symbol[(chr_num, pos)].update(gene_symbol)

        symbol_set.update(gene_symbol)
print(f'symbol_set in GWAS data: {len(symbol_set)}:,')

# 读取 gene_info 文件
# gene to symbol
# 读取NCBI的gene_info.gz文件
gene_info_file = '../../../hg38_data/gene_info.gz'

entrez_to_symbol = {}
symbol_to_entrez = {}
with gzip.open(gene_info_file, 'r') as f:
    f.readline()
    for line in f:
        l = line.decode('utf-8').strip().split('\t')

        tax_id = l[0]
        if tax_id != '9606':
            continue
        entrez = l[1]
        symbol = l[2]

        entrez_to_symbol[entrez] = symbol
        symbol_to_entrez[symbol] = entrez


"""
# 看能通过bedtools注释的symbol召回多少entrez id
# 用 NCBI 的 gene_info.gz 文件

mapping symbol to p and entrez
symbol  entrez   P-value
"""

# mapping symbol to p

symbol_to_most_sig_p = {}

for pos, p in pos_to_most_sig_p.items():
    symbol_set = pos_to_symbol[pos]

    for symbol in symbol_set:
        if symbol_to_most_sig_p.get(symbol):
            if p < symbol_to_most_sig_p[symbol]:
                symbol_to_most_sig_p[symbol] = p
        else:
            symbol_to_most_sig_p[symbol] = p



sort_symbol = sorted(symbol_to_most_sig_p.keys(), key=lambda x: symbol_to_most_sig_p[x])

save_file = 'symbol_entrez_p.txt'
save_count = 0

symbol_recall = set()
symbol_miss = set()

entrez_to_match_symbol ={}
entrez_to_p = {}
with open(save_file, 'w') as wf:
    for symbol in sort_symbol:
        p = symbol_to_most_sig_p[symbol]

        if symbol_to_entrez.get(symbol):
            symbol_recall.add(symbol)

            entrez = symbol_to_entrez[symbol]

            entrez_to_match_symbol[entrez] = symbol
            entrez_to_p[entrez] = p
            save_count += 1

            wf.write(f'{symbol}\t{entrez}\t{p}\n')
        else:
            symbol_miss.add(symbol)

print(f'gene symbol recall (in NCBI gene_info.gz): {len(symbol_recall):,}, '
      f'miss: {len(symbol_miss):,}')


print(f'{save_count:,} symbol-entrez-p saved.')

# 合并AGAC标注文件
save_tag_type = {'CPA', 'MPA', 'Disease', 'Phenotype'}

sentence_file = '../../../../AGAC_tagging_data/AD.pmc.tagging.txt'

MPA_CPA_DISEASE_HPO_save_file = 'MPA.CPA.DIS.HPO.tsv'
# 读 AGAC注释文件
entrez_to_text = defaultdict(set)
entrez_to_text_count = defaultdict(int)
sentence_recall_entrez = set()
sentence_to_tags = defaultdict(set)

MPA_CPA_DIS_HPO_set = set()
with open(sentence_file) as f:
    for line in f:
        l = line.strip().split('\t')
        pmid = l[1]
        sentence = l[2]
        tags = l[3:]
        for _ in tags:
            tag = eval(_)
            sentence_to_tags[ sentence ].add(tag)

            symbol = tag[ 0 ]
            type = tag[ 1 ]
            entrez = tag[ 2 ]

            if type in save_tag_type:
                MPA_CPA_DIS_HPO_set.add((type, symbol))
            if len(tag) == 3:
                continue
            if type == 'Gene':
                sentence_recall_entrez.add(entrez)
                entrez_to_text[entrez].add((pmid, sentence))
                entrez_to_text_count[entrez] += 1

with open(MPA_CPA_DISEASE_HPO_save_file, 'w') as wf:
    wf.write('Type\tPhrase\n')
    for _type, symbol in MPA_CPA_DIS_HPO_set:
        wf.write(f'{_type}-{symbol}\t{symbol}\n')

print(f'sentence_recall_entrez: {len(sentence_recall_entrez)}')


# todo: entrez_to_text Quality Control

save_sentence_count = 10
entrez_to_quality_text = {}

for entrez in entrez_to_text.keys():
    if len(entrez_to_text[entrez]) <= save_sentence_count:
        entrez_to_quality_text[entrez] = entrez_to_text[entrez]
    else:
        sentence_to_quality_tag_count = defaultdict(int)
        for (pmid, sentence) in entrez_to_text[entrez]:
            tags = sentence_to_tags[sentence]

            for tag in tags:
                tag_type = tag[2]
                if tag_type in save_tag_type:
                    sentence_to_quality_tag_count[(pmid, sentence)] += 1
                else:
                    sentence_to_quality_tag_count[(pmid, sentence)] = sentence_to_quality_tag_count[(pmid, sentence)]

        save_sentence_list = list(sorted(sentence_to_quality_tag_count.keys(), key=lambda x: sentence_to_quality_tag_count[x], reverse=True))

        entrez_to_quality_text[entrez] = save_sentence_list[:save_sentence_count]



# 保存 symbol-entrez-sentence 文件
save_file = 'entrez_p_sentence.tsv'
recall_entrez = set()
entrez_sort = sorted(entrez_to_p, key=lambda x:entrez_to_p[x])
with open(save_file, 'w') as wf:
    for entrez in entrez_sort:
        if entrez_to_quality_text.get(entrez):
            recall_entrez.add(entrez)
            p = entrez_to_p[entrez]
            symbol = entrez_to_symbol[entrez]
            wf.write(f'GENE_LINE:\t{symbol}\t{entrez}\t{p}\n')
            for (pmid, sentence) in entrez_to_quality_text[entrez]:
                tag = str(sentence_to_tags[sentence])
                wf.write(f'{pmid}\t{sentence}\t{tag}\n')

print(f'sentence_map_recall_entrez: {len(recall_entrez)}')

