# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 11/09/2021 22:55
@Author: XINZHI YAO
"""
import argparse
import os.path
import re
import time
from collections import defaultdict

from nltk.corpus import stopwords
from string import punctuation

import torch

from transformers import BertModel, BertTokenizer

"""
1.对于每个gene-disease 关联，所有的句子计算平均嵌入
2. 单的的每个短语或者句子计算嵌入
"""



class Bag_DataLoader:

    def __init__(self, input_file: str):

        self.input_file = input_file
        self.entrez_bag = defaultdict(set)
        self.data_size = 0
        self.stopword = stopwords.words('english')
        self.punc = punctuation

        self.stop_word_init()

        if SINGLE_SENTENCE_FILE:
            self.read_single_data()
        else:
            self.read_bag_data()

    def stop_word_init(self):
        self.stopword = stopwords.words('english')

    def read_single_data(self):
        print(f'Load input file: {self.input_file}.')

        with open(self.input_file) as f:
            f.readline()
            for line in f:
                sent_idx, sentence = line.strip().split('\t')

                if TEXT_NORM:
                    sentence = self.normalizeString(sentence)

                self.entrez_bag[sent_idx].add(sentence)
        self.data_size = len(self.entrez_bag)
        print(f'data_size: {self.data_size}')

    def read_bag_data(self):

        entrez = ''
        print(f'Load input file: {self.input_file}.')
        with open(self.input_file) as f:
            for line in f:
                if line.startswith('GENE_LINE:'):
                    l = line.strip().split('\t')
                    symbol, entrez, p = l[1], l[2], float(l[3])
                else:
                    l = line.strip().split('\t')
                    sentence = l[1]

                    if TEXT_NORM:
                        sentence = self.normalizeString(sentence)

                    self.entrez_bag[entrez].add(sentence)
        self.data_size = len(self.entrez_bag)
        print(f'data_size: {self.data_size}')

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(self, s):
        # remove (.*?)
        pattern = re.compile(r'[(].*?[)]', re.S)
        for brackets in re.findall(pattern, s):
            s = s.replace(brackets, ' ')
        # remove stopword
        for sw in self.stopword:
            s = re.sub(r'\b{0}\b'.format(sw), ' ', s)
        # remove punc
        for punc in self.punc:
            s = s.replace(punc, ' ')
        # remove consecutive spaces
        s = re.sub(' +', ' ', s)
        return s

    def __len__(self):
        return len(self.entrez_bag.keys())

def tensor_to_list(_tensor):
    return [str(d) for d in _tensor.detach().numpy().tolist()]

def Last_layer_CLS(phrase_all_hidden_states):
    phrase_embedding = phrase_all_hidden_states[-1:, -1].squeeze()
    # print(phrase_embedding.shape)
    return phrase_embedding

def get_bag_embedding(bag, model, tokenizer):

    sum_embedding = torch.zeros(EMBEDDING_SIZE)

    encoded_input = tokenizer(list(bag), return_tensors='pt', padding=True,
                          truncation=True, max_length=MAX_LEN).to(DEVICE)

    batch_attention_mask = encoded_input['attention_mask']
    output = model(**encoded_input, output_hidden_states=True,
                   output_attentions=True)

    # last_hidden_state = output['last_hidden_state']
    all_hidden_states = output['hidden_states']
    # all_attentions = output['attentions']

    all_hidden_states = torch.stack(all_hidden_states, dim=0)
    all_hidden_states = all_hidden_states.permute(1, 0, 2, 3).cpu()

    for sentence_idx, attention_mask in enumerate(batch_attention_mask):

        sentence_all_hidden_states = all_hidden_states[sentence_idx]

        sentence_embedding = Last_layer_CLS(sentence_all_hidden_states)


        sum_embedding += sentence_embedding

    bag_embedding = tensor_to_list(sum_embedding/len(bag))

    return bag_embedding


def BertForBagEmbedding():

    print(f'model: {MODEL}')

    start_time = time.time()

    print(f'Device: "{DEVICE.type}"')

    print(f'Loading Model and tokenizer from "{MODEL}"')
    model = BertModel.from_pretrained(MODEL).to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained(MODEL)

    DataLoader = Bag_DataLoader(INPUT_FILE)
    entrez_bag_data = DataLoader.entrez_bag
    data_size = DataLoader.data_size

    wf = open(SAVE_FILE, 'w')
    bag_count = 0
    with torch.no_grad():
        for entrez, bag_set in entrez_bag_data.items():
            bag_count += 1

            if len(bag_set) > MAX_BAG_SIZE:
                bag_set = list(bag_set)[:MAX_BAG_SIZE]

            if bag_count % 100 == 0:
                print(f'{bag_count}/{data_size} Bag Processed.')

            # bag embedding
            bag_embedding = get_bag_embedding(bag_set, model, tokenizer)

            embedding_wf = ' '.join(bag_embedding)
            wf.write(f'{entrez}\t{embedding_wf}\n')

    wf.close()
    end_time = time.time()
    print(f'{SAVE_FILE} save done, cost: {end_time-start_time:.2f} s.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='bag_embedding')

    parser.add_argument('--input_file', dest='input_file', required=True)
    parser.add_argument('--output_file', dest='output_file', required=True)

    parser.add_argument('--max_len', dest='max_len', type=int, default=500)
    parser.add_argument('--max_bag_size', dest='max_bag_size', type=int, default=20)
    parser.add_argument('--embedding_size', dest='embedding_size', type=int, default=1024)

    parser.add_argument('--model', dest='model', default='large', choices=['large', 'base'])

    parser.add_argument('--use_cpu', dest='use_cpu', action='store_true', default=False)
    parser.add_argument('--single_sentence_file', dest='single_sentence_file', action='store_false', default=True)

    parser.add_argument('--text_norm', dest='text_norm', action='store_false', default=True)

    args = parser.parse_args()

    MAX_LEN = args.max_len
    MAX_BAG_SIZE = args.max_bag_size

    # large/base bert model 1024
    EMBEDDING_SIZE = args.embedding_size

    if args.model == 'base':
        MODEL = 'dmis-lab/biobert-base-cased-v1.1'
    else:
        MODEL = 'dmis-lab/biobert-large-cased-v1.1'

    USE_CPU = args.use_cpu
    DEVICE = torch.device('cuda' if torch.cuda.is_available() and not USE_CPU else 'cpu')

    # Single Sentence File (Term data) or bag data
    # sentence_id   sentence
    SINGLE_SENTENCE_FILE = args.single_sentence_file

    TEXT_NORM = args.text_norm

    INPUT_FILE = args.input_file

    SAVE_FILE = args.output_file

    BertForBagEmbedding()

