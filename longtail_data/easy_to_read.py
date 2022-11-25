import csv
import os

import numpy as np
import imp
import jsonlines
import json
import copy
import sacrebleu
import argparse
import glob
from collections import defaultdict as ddict
from get_data_utils import All_Data
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import math


class Perplexity:
    def __init__(self,name, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.device = device
        self.model.to(self.device)

    def calculate_perplexity(self,stm):

        tokens = self.tokenizer(stm,return_tensors='pt')
        tokens = tokens.to(self.device)

        loss = self.model(**tokens,labels = tokens['input_ids'],return_dict = True).loss
        loss = loss * len(tokens)
        return math.exp(loss.item())




def check_constraint(cons,generation):
    generation_l = generation.strip().split(' ')

    all_cons = []
    for concepts in cons:
        concepts_state = False
        for word in concepts:
            if word in generation_l or word.capitalize() in generation_l:
                concepts_state = True
                break
        all_cons.append(concepts_state)
    return all(all_cons)



def back_conti_sent(sent, generation, mask = '<extra_id_0>'):
    return sent.replace(mask,generation)


def back_sent(sent, conj_word, generation, mask = '[mask]'):
    mask_index = sent.index(mask)
    head = sent[:mask_index-1]
    assert 'If' in head,'If is not in head'
    
    
    head = head[3:]
    
    tail = sent[mask_index + len(mask) + 2:-1]

    head_low = head[0].lower() + head[1:]

    
    if conj_word == 'and':
        sent = head + ' and ' + generation + ', so ' + tail + '.'
    elif conj_word == 'while':
        sent = head + ' while ' + generation + ', so ' + tail + '.'
    elif conj_word == 'but':
        sent = generation.capitalize() + ' but ' + head_low + ', so ' + tail + '.'
    elif conj_word == 'although':
        sent = 'Although ' + generation + ', ' + head_low + ', so ' + tail + '.'

    return sent




def constraints(data,has_neg=False):
    object2 = data['object2']
    cons = copy.deepcopy(data['constraint'])
    cons['noun'].append(object2)
    
    if has_neg:
        cons['neg'] = ['no']
    return cons
    
def main(args):

    all_data = All_Data()
    ppl = Perplexity('gpt2-large')




    all_dict = all_data.all_data


    
    files = sorted(glob.glob(f'{args.dir}/*.csv'))

    name_dict = {}
    extension = '.csv'
    for name in files:
        name_dict[name.split('/')[-1].replace(extension,'').strip()] = name
    
    for _ in name_dict.keys():
        path = name_dict[_]
        tmp_dict = {}
        with open(path) as f:
            reader = csv.reader(f)
            for line in reader:

                generation_part, id = line[0],line[1]
                tmp_dict[id] = generation_part
        name_dict[_] = tmp_dict



    id_all = set()
    for name in name_dict.keys():
        id_all = id_all | set(list(name_dict[name].keys()))

    
    for name in name_dict.keys():
        id_all = id_all & set(list(name_dict[name].keys()))



    id_all = sorted(list(id_all),key = lambda i : int(i.split('_')[0]))

    nouns_length_dict = {}
    
    for id in id_all:
        nouns_length_dict[id] = len(all_dict[id]['cons_lemma']) - 2 if 'neg' in id else len(all_dict[id]['cons_lemma']) - 1


    id_all = sorted(list(id_all),key = lambda i : nouns_length_dict[i], reverse = True)

    o_path = f'{args.dir}/compare.txt'
    fo = open(o_path,'w')

    
    nl = '\n'
    cons_state_dict = ddict(list)
    ppl_score_dict = ddict(list)
    ppl_score_generation_dict = ddict(list)
    generation_length_dict = ddict(list)

    for id in id_all:

        _ = all_dict[id]
        index = _['index']
        base = _['base']
        sample_conti = _['sample_cont']
        cons = _['cons_lemma']
        inflections = _['cons_inflection']
    

        fo.write(f'{base}  |   {id}')
        fo.write(nl)
        fo.write(f'sample conti: {sample_conti}')
        fo.write(nl)
        fo.write(f'lemmas: {cons}')
        fo.write(nl)
        fo.write('*'*50)
        fo.write(nl)
        fo.write(nl)
        for name in list(name_dict.keys()):
            generations = name_dict[name]
            
            generation = generations[id]
            generation = generation[:-1] if generation.endswith('.') else generation

            cons_state = check_constraint(inflections,generation)
            cons_state_dict[name].append(cons_state)

            original_data = all_dict[id]
            normal_template = original_data['normal_template']

            filled_stm = normal_template.replace('[mask]',generation)

            ppl_score = ppl.calculate_perplexity(filled_stm)
            ppl_score_dict[name].append(ppl_score)


            ppl_score_generation = ppl.calculate_perplexity(generation)
            ppl_score_generation_dict[name].append(ppl_score_generation)

            generation_length_dict[name].append(len(generation.split(' '))/len(inflections))


            fo.write(name)
            fo.write(nl)
            fo.write(filled_stm)
            fo.write(nl)
            fo.write(nl)
            fo.write('*' * 20)
            fo.write(nl)
            fo.write(nl)

        fo.write(nl)
        fo.write(nl)
        fo.write(nl)
    
    fo.write('Ratio for each model that follow the constraints strictly')
    fo.write(nl)
    for name in cons_state_dict.keys():
        cons_state_l = list(cons_state_dict[name])
        fo.write(f'{name}:   {np.sum(cons_state_l)/len(cons_state_l)}')
        fo.write(nl)
    fo.write(nl)
    fo.write(nl)


    fo.write('PPL score for whole statements for each model')
    fo.write(nl)
    for name in ppl_score_dict.keys():
        ppl_score = list(ppl_score_dict[name])
        fo.write(f'{name}:   {np.mean(ppl_score)}')
        fo.write(nl)
    fo.write(nl)
    fo.write(nl)



    fo.write('PPL score for filling part for each model')
    fo.write(nl)
    for name in ppl_score_generation_dict.keys():
        ppl_score_generation = list(ppl_score_generation_dict[name])
        fo.write(f'{name}:   {np.mean(ppl_score_generation)}')
        fo.write(nl)
    fo.write(nl)



    fo.write('Length of generation for each model')
    fo.write(nl)
    for name in generation_length_dict.keys():
        generation_length= list(generation_length_dict[name])
        fo.write(f'{name}:   {np.mean(generation_length)}')
        fo.write(nl)
    fo.write(nl)
    
        
    
    
        
    
            
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',default='./generated_data/property_centric')
    args = parser.parse_args()
    main(args)