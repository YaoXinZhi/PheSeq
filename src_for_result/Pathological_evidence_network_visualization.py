# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 05/05/2022 15:29
@Author: XINZHI YAO
"""
import argparse

"""
This code is used to visualize pathological evidence networks
"""

import os
import gzip
from collections import defaultdict

def read_string_db(string_file: str):
    threshold = 500
    print(f'reading {string_file}')
    gene_pair_set = set()
    with gzip.open(string_file) as f:
        f.readline()
        for line in f:
            l = line.decode('utf-8').strip().split()
            #print(l)
            score = int(l[-1])
            if (l[0], l[1]) not in gene_pair_set and (l[1], l[0]) not in gene_pair_set:
                if score >= threshold:
                    gene_pair_set.add((l[0], l[1]))
    print(f'Gene pair count from STRING: {len(gene_pair_set):,}')
    return gene_pair_set

def get_to_n_for_vis(report_file: str, topn:int):
    line_count = 0

    top_gene_list = []
    selected_go_to_term = {}
    entrez_to_go = defaultdict(set)
    go_to_entrez = defaultdict(set)
    with open(report_file) as f:
        f.readline()
        for line in f:
            l = line.strip().split('\t')

            entrez = l[0]

            top_gene_list.append(entrez)

            tags = l[-1].split('  ')

            for tag in tags:
                if tag.startswith('GO:'):
                    go_id, official, mention = tag.split('|')

                    entrez_to_go[entrez].add(go_id)

                    go_to_entrez[go_id].add(entrez)

                    selected_go_to_term[go_id] = official
            line_count += 1
            if line_count >= topn:
                break

    top_gene_set = set(top_gene_list)
    return top_gene_list, top_gene_set, selected_go_to_term

def read_gene_info(gene_info_file: str, tax: str):
    entrez_to_symbol = {}
    symbol_to_entrez = {}
    with gzip.open(gene_info_file, 'r') as f:
        f.readline()
        for line in f:
            l = line.decode('utf-8').strip().split('\t')

            tax_id = l[ 0 ]
            if tax_id != tax:
                continue
            entrez = l[ 1 ]
            symbol = l[ 2 ]

            entrez_to_symbol[ entrez ] = symbol
            symbol_to_entrez[ symbol ] = entrez

    return entrez_to_symbol, symbol_to_entrez

def read_obo_file(obo_file: str):
    go_id_to_term = {}
    go_alt_id_to_term = {}
    go_to_type = {}
    with open(obo_file) as f:
        for line in f:
            l = line.strip().split()
            if line.startswith('id:'):
                go_id = l[ 1 ]
            if line.startswith('name:'):
                go_term = ' '.join(l[ 1: ])
                go_id_to_term[ go_id ] = go_term
            if line.startswith('namespace:'):
                go_type = l[ 1 ]
                go_to_type[ go_id ] = go_type
            if line.startswith('alt_id:'):
                go_alt_id = l[ 1 ]
                go_alt_id_to_term[ go_alt_id ] = go_term

    return go_id_to_term, go_alt_id_to_term, go_to_type

def read_gene_to_ensembl_file(gene_to_ensembl_file: str, tax: str):
    ensembl_to_entrez = {}
    with open(gene_to_ensembl_file) as f:
        for line in f:
            l = line.strip().split('\t')

            tax_id = l[ 0 ]
            if tax_id != tax:
                continue

            entrez = l[ 1 ]
            ensembl = l[ -1 ].split('.')[ 0 ]

            ensembl_to_entrez[ ensembl ] = entrez
    print(f'ensembl_to_entrez: {len(ensembl_to_entrez):,}')
    return ensembl_to_entrez

def string_ensembel_to_entrez(string_gene_pair_set: set, ensembl_to_entrez: dict):
    string_entrez_pair_set = set()
    for gene1, gene2 in string_gene_pair_set:

        ensembl_1 = gene1.split('.')[ 1 ]
        ensembl_2 = gene2.split('.')[ 1 ]

        if ensembl_to_entrez.get(ensembl_1) and ensembl_to_entrez.get(ensembl_2):
            entrez_1 = ensembl_to_entrez[ ensembl_1 ]
            entrez_2 = ensembl_to_entrez[ ensembl_2 ]
            string_entrez_pair_set.add((entrez_1, entrez_2))
    return string_entrez_pair_set

def read_seq_gene_file(seq_gene_file: str):

    seq_gene_set = set()
    with open(seq_gene_file) as f:
        for line in f:
            l = line.strip()
            seq_gene_set.add(l)
    return seq_gene_set

def read_evidence_file(evidence_file: str, top_gene_set: set):
    entrez_to_go = defaultdict(set)
    entrez_to_hpo = defaultdict(set)

    entrez_to_go_evi = defaultdict(set)
    entrez_to_hpo_evi = defaultdict(set)
    with open(evidence_file) as f:

        for line in f:
            l = line.strip().split('\t')
            pmid = l[1]
            sent = l[2]
            tag_list = [eval(tag) for tag in l[3:]]

            gene_set = set()
            for tag in tag_list:
                if len(tag) == 4 and tag[1] == 'Gene':
                    gene_mention = l[0]
                    entrez = l[2]
                    if not entrez in top_gene_set:
                        continue
                    gene_set.add((gene_mention, entrez))

            if not gene_set:
                continue

            for tag in tag_list:
                if len(tag) == 4 and tag[2].startswith('GO'):
                    go_id = l[2]
                    go_mention = l[0]
                    for gene_mention, entrez in gene_set:
                        entrez_to_go_evi[entrez].add((gene_mention, go_id, go_mention, pmid, sent))
                        entrez_to_go[entrez].add(go_id)

                if len(tag) == 4 and tag[1] == 'Phenotype':
                    hpo_id = l[2]
                    hpo_mention = l[0]
                    for gene_mention, entrez in gene_set:
                        entrez_to_hpo_evi[entrez].add((gene_mention, hpo_id, hpo_mention, pmid, sent))
                        entrez_to_hpo[entrez].add(hpo_id)

    return entrez_to_go_evi, entrez_to_hpo_evi, entrez_to_go, entrez_to_hpo


def process_node_info(top_gene_set: set, gwas_gene_set: set,
                      entrez_to_symbol: dict, selected_go_to_term: dict,
                      go_id_to_term: dict):
    gene_node_size = 60
    go_node_size = 40

    overlapping_gene_set = top_gene_set&gwas_gene_set

    our_only_gene_set = top_gene_set - gwas_gene_set

    gwas_only_gene_set = gwas_gene_set - top_gene_set

    draw_lit_gene_set = set()
    draw_gwas_gene_set = set()
    draw_both_gene_set = set()

    gene_go_node_list = []


    # GWAS node with blue
    for entrez in gwas_only_gene_set:
        draw_gwas_gene_set.add(entrez)

        gene_go_node_list.append({'name': entrez_to_symbol[ entrez ],
                                  'value': f'{entrez}:{entrez_to_symbol[ entrez ]}',
                                  'symbolSize': gene_node_size,
                                  'draggable': 'True',
                                  'categories': 'Gene',
                                  'label': {'show': 'true',
                                            'position': 'inside',
                                            'fontWeight': 'bolder', },
                                  'category': 'Gene',
                                  'itemStyle': {
                                      'color': '#00a8e1',
                                  },
                                  })
    # EMFAS node with red
    for entrez in our_only_gene_set:
        draw_lit_gene_set.add(entrez)

        gene_go_node_list.append({'name': entrez_to_symbol[ entrez ],
                                  'value': f'{entrez}:{entrez_to_symbol[ entrez ]}',
                                  'symbolSize': gene_node_size,
                                  'draggable': 'True',
                                  'categories': 'Gene',
                                  'label': {'show': 'true',
                                            'position': 'inside',
                                            'fontWeight': 'bolder', },
                                  'category': 'Gene',
                                  'itemStyle': {
                                      'color': '#e30039',
                                  },
                                  })

    # overlapping gene node with blue and red cycle
    for entrez in overlapping_gene_set:
        draw_both_gene_set.add(entrez)
        gene_go_node_list.append({'name': entrez_to_symbol[ entrez ],
                                  'value': f'{entrez}:{entrez_to_symbol[ entrez ]}',
                                  'symbolSize': gene_node_size,
                                  'draggable': 'True',
                                  'categories': 'Gene',
                                  'label': {'show': 'true',
                                            'position': 'inside',
                                            'fontWeight': 'bolder', },
                                  'category': 'Gene',
                                  'itemStyle': {
                                      'color': '#00a8e1',
                                      'borderColor': '#e30039',
                                      'borderWidth': 6,
                                      'borderType': 'solid',
                                  },
                                  })

    # GO node with green
    for go_id, term in selected_go_to_term.items():
        if go_id_to_term.get(go_id):
            official_name = go_id_to_term[go_id]
        else:
            continue
        gene_go_node_list.append({'name': official_name,
                                  'value': f'{go_id}:{official_name}',
                                  'symbolSize': go_node_size,
                                  'draggable': 'True',
                                  'categories': 'Gene',
                                  'label': {'show': 'true',
                                            'position': 'inside',
                                            'fontWeight': 'bolder', },
                                  'category': 'Gene',
                                  'itemStyle': {
                                      'color': '#99cc00',
                                  },
                                  })
    node_str = f'var data={gene_go_node_list}'

    return node_str

def get_offset(token: str, sentence: str):

    offset_set = set()
    start = 0
    while True:
        token_start = sentence.find(token, start)

        if token_start == -1:
            break
        token_end = token_start + len(token)
        start = token_end

        offset_set.add((token_start, token_end))
    return offset_set


def get_html_rich(source_token: str, target_token: str,
                  source_type: str, target_type: str,
                  sentence: str, ):
    type_to_html_color = {
        'gene': '#e30039',
        'GO': '#99cc00',
        'HPO': '#00994e',
    }

    character_list = [ s for s in sentence ]

    source_offset_set = get_offset(source_token, sentence)

    target_offset_set = get_offset(target_token, sentence)

    offset_dic = {}
    offset_key_to_type = {}
    for source_offset in source_offset_set:
        _key = len(offset_dic)
        offset_dic[ _key ] = source_offset

        offset_key_to_type[ _key ] = source_type

    for target_offset in target_offset_set:
        _key = len(offset_dic)
        offset_dic[ _key ] = target_offset

        offset_key_to_type[ _key ] = target_type

    offset_sorted = sorted(offset_dic.keys(), key=lambda x: offset_dic[ x ][ 0 ],
                           reverse=True)
    for token_key in offset_sorted:
        token_type = offset_key_to_type[ token_key ]
        offset = offset_dic[ token_key ]
        # html rich text
        character_list.insert(offset[ 1 ], '</b></font>')
        character_list.insert(offset[ 0 ],
                              f'<font color="white" style="background:{type_to_html_color[ token_type ]}"><b>')

    rich_sent = ''.join(character_list)
    return rich_sent

def save_rich_text(entrez_to_term_evi: dict, rich_save_file: str):
    max_length = 1200

    with open(rich_save_file, 'w') as wf:
        wf.write('Gene ID\tGene mention\tHPO ID\tHPO mention\tRich text\tPMID\n')
        for gene_id, evidence_set in entrez_to_term_evi.items():
            for evidence in evidence_set:

                gene_mention, hpo_id, hpo_mention, pmid, sentence = evidence
                if len(gene_mention) < 3 or len(hpo_mention) < 3:
                    continue

                rich_text = get_html_rich(gene_mention, hpo_mention, 'gene', 'GO', sentence)

                if max_length < len(rich_text):
                    continue

                wf.write(f'{gene_id}\t{gene_mention}\t{hpo_id}\t{hpo_mention}\t{rich_text}\t{pmid}\n')

def padding(list_a: list, list_b: list):
    if len(list_a) < len(list_b):
        list_a.extend(['[pad]']*(len(list_b)-len(list_a)))
        return list_a, list_b
    if len(list_b) < len(list_a):
        list_b.extend(['[pad]']*(len(list_a)-len(list_b)))
        return list_a, list_b
    return list_a, list_b

def pretty_html_text(rich_text: str, line_token: int = 4):
    text_split = re.split(r'<.*?>', rich_text)

    tag_list = re.findall(r'<.*?>', rich_text)

    text_split, tag_list = padding(text_split, tag_list)

    pretty_doc = [ ]
    batch_token = [ ]
    token_list = [ ]
    text_token = False

    token_count = 0
    for text, tag in zip(text_split, tag_list):
        for token in text.split(' '):

            token_count += 1
            if token != '[pad]' and token != '':
                batch_token.append(token)
                if text_token:
                    token_list.append(token)

                if token_count % line_token == 0 and len(token_list) != 0:
                    batch_token.append('<br/>')
        pretty_doc.append(' '.join(batch_token))
        batch_token = [ ]
        if tag != '[pad]':
            pretty_doc.append(tag)
            text_token = True
    if '' in pretty_doc:
        pretty_doc.remove('')

    pretty_doc = ' '.join(pretty_doc)
    pretty_doc = re.sub(' +', ' ', pretty_doc)
    pretty_doc = pretty_doc.replace(' </font>', '</font>')
    pretty_doc = pretty_doc.replace(' </b>', '</b>')
    pretty_doc = pretty_doc.replace(' ,', ',')
    pretty_doc = pretty_doc.replace(' .', '.')

    return pretty_doc

def read_rich_file(rich_file: str, filter_key: str):
    gene_hpo_id_to_example = {}

    with open(rich_file) as f:
        f.readline()
        for line in f:
            gene_id, gene_mention, hpo_id, hpo_mention, sent, pmid = line.strip().split('\t')

            term_key = (gene_id, hpo_id)
            if not gene_hpo_id_to_example.get(term_key):
                gene_hpo_id_to_example[ term_key ] = (pmid, sent)
            else:
                if filter_key in sent:
                    gene_hpo_id_to_example[ term_key ] = (pmid, sent)
    return gene_term_id_to_example

def process_edge_info(top_gene_set: set, gwas_gene_set: set,
                 string_entrez_pair_set: set,
                 entrez_to_term: str,
                 gene_term_id_to_example: dict):

    string_edges = [ ]

    overlapping_gene_set = top_gene_set & gwas_gene_set

    our_only_gene_set = top_gene_set - gwas_gene_set

    gwas_only_gene_set = gwas_gene_set - top_gene_set

    total_gene_node_set = top_gene_set | gwas_gene_set
    # edges from STRING
    saved_edge_set = set()
    for source_entrez in total_gene_node_set:
        for target_entrez in total_gene_node_set:
            if (source_entrez, target_entrez) in string_entrez_pair_set \
                    or (target_entrez, source_entrez) in string_entrez_pair_set:
                if (source_entrez, target_entrez) in saved_edge_set \
                        or (target_entrez, source_entrez) in saved_edge_set:
                    continue

                source = entrez_to_symbol[ source_entrez ]
                target = entrez_to_symbol[ target_entrez ]
                saved_edge_set.add((source_entrez, target_entrez))

                string_edges.append({'source': source,
                                     'target': target,
                                     'value': f'Gene-Gene Interaction from STRING.',
                                     'lineStyle': {'color': '#0F0F0F'},
                                     'emphasis': {
                                         'lineStyle': {'width': 3},
                                     },
                                     })

    # edges for gene-GO
    entrez_to_selected_go_count = defaultdict(int)
    gene_go_edge_count = 0
    miss_count = 0
    miss_set = set()
    for entrez in set(overlapping_gene_set | our_only_gene_set | gwas_only_gene_set):
        go_set = entrez_to_term[ entrez ]
        entrez_to_selected_go_count[ entrez ] = 0
        for go_id in go_set:
            entrez_to_selected_go_count[ entrez ] += 1
            source = entrez_to_symbol[ entrez ]

            if go_id_to_term.get(go_id):
                target = go_id_to_term[ go_id ]
            else:
                target = go_alt_id_to_term[ go_id ]

            if not gene_term_id_to_example.get((entrez, go_id)):
                miss_count += 1

                miss_set.add((entrez, go_id))
                continue
            else:
                pmid, rich_text = gene_term_id_to_example[ (entrez, go_id) ]

            gene_go_edge_count += 1
            string_edges.append({'source': source,
                                 'target': target,
                                 'value': f'{pmid}: {rich_text}',
                                 'lineStyle': {'color': '#458B00'},
                                 'emphasis': {
                                     'lineStyle': {'width': 3},
                                 },
                                 })
    edge_str = f'var links={string_edges}'
    return edge_str

def generate_html(node_str: str, edge_str: str, save_file: str,):
    html_head = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">

    <script type="text/javascript" src="https://assets.pyecharts.org/assets/echarts.min.js"></script>    


    <!-- <link rel="stylesheet" type="text/css" href="css/style.css"> -->
    <style>
        .container {
            margin-left:-100px;
        }
        .container2 {
            margin-left: 100px;
            margin-right: -300px;
        }
        thead {
            background-color: rgba(244, 67, 54, 0.85);
            color: #fff;
        }
        mark {
            background-color: #f4f57a;
            padding: 0;
        }
        button.dt-button, div.dt-button, a.dt-button {
            padding: 0.3em 0.3em;
        }
    </style>
    <title>Beautiful Tables</title>
    </head>
    <body>

    <div id="cc51ebcfed2642fbbc21ce3b95b1f0a7" class="chart-container" style="width:1100px; height:1000px;"></div>
    <script>
	
    function my$(id){
    return document.getElementById(id);
    }
    function figure$(id,data,link){
	
        var MyChart = echarts.init(document.getElementById(id), 'light', {renderer: 'canvas'});
		
		var option = {
					
				
				"tooltip": {
						"show": true,
						
						'position': {left: '5%', top: '10%'},
						"trigger": "item",
						"triggerOn": "click",
						"enterable": true,
						"alwaysShowContent": true,
						"renderMode": 'html',
						
						"width": 20,
						"overflow": "breakAll",
						
						formatter: function(param) {
							var text = ''
							text += param.name
							text += '<br/>'
							text += param.value
							return text
						},
			
						"textStyle":{
							'color':"#333",
							"fontStyle": 'italic',
							"fontWeight": 'normal',
							"fontFamily": 'Courier New',
							//"fontFamily": 'Arial',
							"fontSize": 20,
							"width": 30,
							"overflow": 'break',
							"rich": {
							"protein":{"color": "red"},
							},
						},
						 extraCssText:'width:800px; white-space:pre-wrap'
						},		

				"textStyle":{
					"color": "#000000",
					"fontStyle": 'italic',
					"fontWeight": 'normal',
					"fontFamily": 'Courier New',
					"fontSize": 20,},
		
		
				"toolbox": {
					"show": true,
					"itemSize": 15,
					"itemGap": 16,
					"right": '5%',
					"top": '5%',

					"feature": {
						"saveAsImage": {
							"show": true,
							"type": 'png',
							"title": "Save as image.",

						},
						"restore": {
							"show": true,
							"title": "Revert",
							},
						"dataView": {
							"show": true,
							"title": "Show Data",
							"readOnly": true,
						}}},
	
		"animation": true,
		"animationThreshold": 2000,
		"animationDuration": 1000,
		"animationEasing": "quinticlnOut",
		"animationDelay": 0,
		"animationDurationUpdate": 300,
		"animationEasingUpdate": "cubicOut",
		"animationDelayUpdate": 0,

		
		"color": [
					"#FCCF73",
                    "#B9D92C",

					],
		"series": [
			{
				"type": "graph",
				"layout": "force",
				"symbolSize": 10,
				"nodeScaleRatio": 0.6,
				"circular": {
					"rotateLabel": true
				},
				"force": {
					"repulsion": 1000,
					"edgeLength": [50, 300],
					"gravity": 0.001,
					"friction": 0.1
				},
				"label": {
					"show": true,
					"position": {left: 10, top: '10%'},
					"margin": 8
				},
				"lineStyle": {
					"show": true,
					"width": 2,
					"opacity": 1,
					"curveness": 0.15,
					"type": "solid"
				},
				"roam": true,
				"draggable": true,
				// show adjacency node
				"focusNodeAdjacency": true,
				"data": data,
				"categories": [
					{
						"name": "Gene"
					},
					{
						"name": "Phenotype"
					},
					{
						"name": "Protein"
					},
					{
						"name": "Enzyme"
					},
					{
						"name": "Pathway"
					},
					{
						"name": "Interaction"
					},
					{
						"name": "CPA"
					},
					{
						"name": "MPA"
					},
					{
						"name": "Var"
					}
				],
				"edgeLabel": {
					"show": false,
					"position": 'top',
					"margin": 8
				},
				"edgeSymbol": [
					null,
					null
				],
				"edgeSymbolSize": [10, 10],
				"links": link
			}
		],
		"legend":
			{
				"data": [
					"Gene",
					"Phenotype",

				],
			"selected": {
					"CPA": true,
					"Enzyme": true,
					"Interaction": true,
					"MPA": true,
					"pathway": true,
					"PosReg": true,
					"Protein": true,
					"Gene": true
				},

				"show": false,
				"padding": 5,
				"itemGap": 20,

				"itemWidth": 30,
				"itemHeight": 25,
				"left": '5%',
				"top": '5%',
				"itemStyle": {
					"borderColor": '#000000',
					"borderWidth": 1.6,
				},
				"textStyle":{
				'color':"#000000",
				"fontStyle": 'italic',
				"fontWeight": 'normal',
				"fontFamily": 'Courier New',
				"fontSize": 16,},
			},
			
	"title": [
			{
				"padding": 5,
				"itemGap": 10
			}
            ],
        };
        
                MyChart.setOption(option);  
    }
        """

    html_tail= """figure$('cc51ebcfed2642fbbc21ce3b95b1f0a7',data,links);
    </script>
    """

    with open(save_file, 'wf') as wf:
        wf.write(f'{html_head}\n')
        wf.write(f'{node_str}\n')
        wf.write(f'{edge_str}\n')
        wf.write(f'{html_tail}\n')

    print(f'{save_file} save done.')



def main():

    parser = argparse.ArgumentParser(description='Pathological evidence network visualization.')

    # report file
    parser.add_argument('--report_file: str', dest='report_file',
                        required=True)

    parser.add_argument('--evidence_file', dest='evidence_file',
                        required=True,
                        help='evidence file for disease.')

    parser.add_argument('--save_file', dest='save_file',
                        required=True,
                        help='Saved web page file, the suffix must be ".html"')

    parser.add_argument('--topn', dest='topn',
                        required=False, default=50,
                        help='Top n genes for visualization.')

    parser.add_argument('--tax_id', dest='tax_id',
                        required=False, default='9606',
                        help='tax id for species of interest.')

    parser.add_argument('--sequence_analysis_gene_file', dest='sequence_analysis_gene_file',
                        required=False, default='',
                        help='Significant gene file from sequence analysis, containing one column, each Entrez id per row')

    parser.add_argument('--add_STRING', dest='add_STRING',
                        action='store_true', default=False)

    parser.add_argument('--filter_keyword', dest='filter_keyword',
                        required=False, default='',
                        help='keyword used to filter evidence example.')


    args = parser.parse_args()

    string_entrez_pair_set = set()
    if args.add_STRING:
        string_file = '../data/9606.protein.links.v11.5.txt.gz'
        gene_to_ensemble_file = '../data/gene2ensembl.gz'
        if os.path.exists(string_file):
            string_gene_pair_set = read_string_db(string_file)
        else:
            raise ValueError('STRING file is not in "../data/", please download the corresponding file from https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz and place it in "../data/"')

        if os.path.exists(gene_to_ensemble_file):
            ensemble_to_entrez = read_gene_to_ensembl_file(gene_to_ensemble_file, args.tax_id)
        else:
            raise ValueError(
                'STRING file is not in "../data/", please download the corresponding file from https://ftp.ncbi.nih.gov/gene/DATA/gene2ensembl.gz and place it in "../data/"')

        string_entrez_pair_set = string_ensembel_to_entrez(string_gene_pair_set, ensemble_to_entrez)

    top_gene_list, top_gene_set, selected_go_to_term = get_to_n_for_vis(args.report_file, args.topn)

    gene_info_file = '../data/gene_info.gz'
    if os.path.exists(gene_info_file):
        entrez_to_symbol, symbol_to_entrez = read_gene_info(gene_info_file, args.tax_id)
    else:
        raise ValueError('gene_info.gz file is not in "../data/", please download the corresponding file from https://ftp.ncbi.nih.gov/gene/DATA/gene_info.gz and place it in "../data/"')

    go_file = '../data/go.obo'
    if os.path.exists(go_file):
        go_id_to_term, go_alt_id_to_term, go_to_type = read_obo_file(go_file)
    else:
        raise ValueError('gene_info.gz file is not in "../data/", please download the corresponding file from http://purl.obolibrary.org/obo/go/go-basic.obo and place it in "../data/"')

    if args.sequence_analysis_gene_file:
        gwas_gene_set = read_seq_gene_file(args.sequence_analysis_gene_file)
    else:
        gwas_gene_set = set()


    entrez_to_go_evi, entrez_to_hpo_evi, entrez_to_go, entrez_to_hpo = read_evidence_file(args.evidence_file, top_gene_set)

    temp_dir = '../temp'

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    rich_file = f'{temp_dir}/rich.txt'

    gene_term_id_to_example = read_rich_file(rich_file, args.filter_keyword)


    node_str = process_node_info(top_gene_set, gwas_gene_set, entrez_to_symbol,
                                 selected_go_to_term, go_id_to_term)


    edge_str = process_edge_info(top_gene_set, gwas_gene_set,
                                 string_entrez_pair_set, entrez_to_go,
                                 gene_term_id_to_example)

    generate_html(node_str, edge_str, args.save_file)


if __name__ == '__main__':
    main()
