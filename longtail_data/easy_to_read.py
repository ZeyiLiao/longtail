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




    all_dict = all_data.all_data

    # n_show = 20
    # n_1 = 7
    # n_2 = 10
    # n_3 = 3
    
    # cons_dict = ddict(list)

    # for key in all_dict.keys():
    #     data = all_dict[key]
    #     index = data['index']
    #     id = data['id']

    #     lemma_length = len(data['cons_lemma']) - 2 if 'neg' in id else len(data['cons_lemma']) - 1

    #     if index not in cons_dict[lemma_length]:
    #         cons_dict[lemma_length].append(index)

    #     if len(cons_dict[1]) > n_1:
    #         cons_dict[1] = cons_dict[1][:n_1]

    #     if len(cons_dict[2]) > n_2:
    #         cons_dict[2] = cons_dict[2][:n_2]

    #     if len(cons_dict[3]) > n_3:
    #         cons_dict[3] = cons_dict[3][:n_3]

    #     if cons_dict[1] + cons_dict[2] + cons_dict[3] ==10:
    #         break

    # cons_dict = dict(cons_dict)
    # needed_index =[cons_dict[key] for key in list(cons_dict.keys()) if key in [1,2,3]]
    # needed_index = [b for a in needed_index for b in a]




    
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

    for id in id_all:

        _ = all_dict[id]
        index = _['index']
        base = _['base']
        sample_conti = _['sample_cont']
        cons = _['cons_lemma']
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

            original_data = all_dict[id]
            normal_template = original_data['normal_template']

            fill_base = normal_template.replace('[mask]',generation)
            fo.write(name)
            fo.write(nl)
            fo.write(fill_base)
            fo.write(nl)
            fo.write(nl)
            fo.write('*' * 20)
            fo.write(nl)
            fo.write(nl)

        fo.write(nl)
        fo.write(nl)
        fo.write(nl)
            
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',default='./generated_data/property_centric')
    args = parser.parse_args()
    main(args)