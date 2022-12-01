import csv
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '9'

import numpy as np
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


def check_extra_neg(cons,generation):
    generation_l = generation.strip().split(' ')

    extra_neg = False
    if cons[-1] != ['no', 'not']:
        neg_l = ["no", "not"]
        for neg_word in neg_l:
            if neg_word in generation_l:
                extra_neg = True
                break

    return extra_neg


def check_constraint_for_mul_words(words,generation_l):

    words_l = words.split(' ')
    for i in range(len(generation_l)):
        if (i + len(words_l) - 1) > (len(generation_l)-1):
            return False
        
        tmp = [generation_l[j] for j in range(i,i+len(words_l))]
        tmp_str = ' '.join(tmp)
        if tmp_str == words:
            return True
        




def check_constraint(cons,generation):
    generation_l = generation.strip().split(' ')


    all_cons = []
    for concepts in cons:
        concepts_state = False
        for word in concepts:

            if len(word.split(' ')) > 1:
                concepts_state = check_constraint_for_mul_words(word,generation_l)
                if concepts_state:
                    break
            else:
                if word in generation_l or word.capitalize() in generation_l:
                    concepts_state = True
                    break

        all_cons.append(concepts_state)
    cons_result = all(all_cons)


    return cons_result








def constraints(data,has_neg=False):
    object2 = data['object2']
    cons = copy.deepcopy(data['constraint'])
    cons['noun'].append(object2)
    
    if has_neg:
        cons['neg'] = ['no']
    return cons

    
def main(args):


    name_sort_dict = {}
    name_sort_dict['base'] = 0
    name_sort_dict['medium'] = 1
    name_sort_dict['large'] = 2
    name_sort_dict['xl'] = 3
    name_sort_dict['6b'] = 4
    name_sort_dict['20b'] = 5

    def for_sort(name):
        for key in list(name_sort_dict.keys()):
            if key in name:
                score = name_sort_dict.get(key,None)
                if 'vanilla' in name:
                    score += len(name_sort_dict.keys())
                return score
        else:
            return -1


    
    files = sorted(glob.glob(f'{args.dir}/*.csv'),key = lambda i : for_sort(i))



    all_data = All_Data('/home/zeyi/longtail/property_centric_process/pilot_all_data.pkl')
    ppl = Perplexity('gpt2-xl')
    all_dict = all_data.all_data

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


    # _name_dict = {}
    # _name_dict['wo_m_gptj_6b_vanilla'] = name_dict['wo_m_gptj_6b_vanilla']
    # name_dict = _name_dict
    
    
    
    id_all = set()
    for name in name_dict.keys():
        id_all = id_all | set(list(name_dict[name].keys()))

    
    for name in name_dict.keys():
        id_all = id_all & set(list(name_dict[name].keys()))

    id_all.remove('1071_and')
    id_all.remove('1071_and_neg')



    id_all = sorted(list(id_all),key = lambda i : int(i.split('_')[0]))


    nouns_length_dict = {}
    
    for id in id_all:
        nouns_length_dict[id] = len(all_dict[id]['cons_lemma']) - 2 if 'neg' in id else len(all_dict[id]['cons_lemma']) - 1


    id_all = sorted(list(id_all),key = lambda i : nouns_length_dict[i], reverse = True)

    o_path = f'{args.dir}/compare.txt'
    fo = open(o_path,'w')

    
    nl = '\n'
    cons_state_dict = ddict(list)
    extra_neg_dict = ddict(list)
    ppl_score_dict = ddict(list)
    ppl_score_generation_dict = ddict(list)
    generation_length_dict = ddict(list)
    type_breakdown_dict = ddict(list)
    jsonl_o = []

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
            generation = generation.replace('"','').strip()

            cons_state = check_constraint(inflections,generation)
            extra_neg = check_extra_neg(inflections,generation)

            cons_state_dict[name].append(cons_state)

            extra_neg_dict[name].append(extra_neg)


            original_data = all_dict[id]
            normal_template = original_data['normal_template']
            if len(generation.split(' ')) == 1:
                generation = '[Only one token which is a bad case]'
            filled_stm = normal_template.replace('[mask]',generation)
            _[name] = filled_stm
            

            ppl_score = ppl.calculate_perplexity(filled_stm)
            ppl_score_dict[name].append(ppl_score)


            ppl_score_generation = ppl.calculate_perplexity(generation)
            if np.isnan(ppl_score_generation):
                breakpoint()

            ppl_score_generation_dict[name].append(ppl_score_generation)

            generation_length_dict[name].append(len(generation.split(' '))/len(inflections))

            
            type_name = 'neg' if 'neg' in id else 'base'
            if 'vanilla' not in name:
                type_breakdown_dict[type_name].append(ppl_score)


            if not cons_state:
                ppl_score_dict[name].pop()
                ppl_score_generation_dict[name].pop()
                generation_length_dict[name].pop()
                if 'vanilla' not in name:
                    type_breakdown_dict[type_name].pop()
                extra_neg_dict[name].pop()




            fo.write(name)
            fo.write(nl)
            fo.write(filled_stm)
            fo.write(nl)
            fo.write(nl)
            fo.write('*' * 20)
            fo.write(nl)
            fo.write(nl)
        jsonl_o.append(_)

        fo.write(nl)
        fo.write(nl)
        fo.write(nl)

    with jsonlines.open('./for_turk.jsonl','w') as f:
        f.write_all(jsonl_o)

    
    fo.write(nl)
    fo.write(nl)
    fo.write('Ratio for each model that follows the constraints strictly')
    fo.write(nl)
    for name in cons_state_dict.keys():
        cons_state_l = list(cons_state_dict[name])
        fo.write(f'{name}:   {np.sum(cons_state_l)/len(cons_state_l)}')
        fo.write(nl)
    fo.write(nl)
    fo.write(nl)


    fo.write('PPL score for whole statements')
    fo.write(nl)
    for name in ppl_score_dict.keys():
        ppl_score = list(ppl_score_dict[name])
        fo.write(f'{name}:   {np.mean(ppl_score)}')
        fo.write(nl)
    fo.write(nl)
    fo.write(nl)



    fo.write('PPL score for filling part')
    fo.write(nl)
    for name in ppl_score_generation_dict.keys():
        ppl_score_generation = list(ppl_score_generation_dict[name])
        fo.write(f'{name}:   {np.mean(ppl_score_generation)}')
        fo.write(nl)
    fo.write(nl)
    fo.write(nl)
    fo.write(nl)



    fo.write('Extra Length of generation for each model [(generation length)/ # of constraints)]')
    fo.write(nl)
    for name in generation_length_dict.keys():
        generation_length= list(generation_length_dict[name])
        fo.write(f'{name}:   {np.mean(generation_length)}')
        fo.write(nl)
    fo.write(nl)
    fo.write(nl)
    fo.write(nl)


    fo.write('Extra negation ratio')
    fo.write(nl)
    for name in extra_neg_dict.keys():
        extra_neg= list(extra_neg_dict[name])
        fo.write(f'{name}:   {np.mean(extra_neg)}')
        fo.write(nl)
    fo.write(nl)
    fo.write(nl)
    fo.write(nl)





    fo.write('PPl score for each breakdown')
    fo.write(nl)
    for name in type_breakdown_dict.keys():
        type_breakdown= list(type_breakdown_dict[name])
        fo.write(f'{name}:   {np.mean(type_breakdown)}')
        fo.write(nl)
    fo.write(nl)
    fo.write(nl)
    fo.write(nl)
    
        
    
    
        
    
            
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',default='./generated_data/property_centric')
    args = parser.parse_args()
    main(args)