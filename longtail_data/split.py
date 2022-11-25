from curses import nl
import json
import random
import numpy as np
from pathlib import Path
import os
import jsonlines
import argparse
from sklearn.model_selection import train_test_split
from get_data_utils import All_Data


np.random.seed(42)


def change_format(x):
    cons_string = ''
    for tmp in x:
        tmp = '[' + ', '.join(tmp) + ']'
        cons_string += str(tmp) + ', '
    cons_string = cons_string[:-2]
    
    return cons_string

class Process:
    def __init__(self,all_dict,mask = '[mask]'):
        self.all_dict = all_dict
        self.mask = mask

    def process_format(self,data_type,generations,ids):
        inputs = []
        outputs = []

        for id, generation in zip(ids,generations):
            original_line = self.all_dict[id]
            cons = original_line['cons_lemma']
            cons = change_format(cons)
            conti_template = original_line['conti_template']

            if data_type == 'w_m_t5':
                input = f"Constraints : {cons} ; Input : {conti_template.replace(self.mask,'<extra_id_0>')} ; Output:"
                output = f"<extra_id_0> {generation} <extra_id_1>"

            elif data_type == 'wo_m_t5':
                input = f"Constraints : {cons} ; Input : {conti_template.replace(self.mask + '.','')}"
                output = generation

            elif data_type == 'wo_m_gpt2':
                input = f"Constraints : {cons} ; Input : {conti_template.replace(self.mask,generation)}"
                output = f"Constraints : {cons} ; Input : {conti_template.replace(self.mask,generation)}"

            else:
                raise NotImplementedError

            input = input.strip()
            output = output.strip()

            inputs.append(input)
            outputs.append(output)

        return inputs,outputs


def split_all(dir_name,X,Y,split):
    Path(f'{dir_name}/{split}').mkdir(exist_ok=True,parents=True)
    fo = open(f'{dir_name}/{split}/data.json','w')
    nl = '\n'

    for x,y in zip(X,Y):
        tmp = {}
        tmp['input'] = x
        tmp['output'] = y
        # _tmp = {}
        # _tmp['translation'] = tmp
        json.dump(tmp,fo)
        fo.write(nl)


def main(args):

    all_data = All_Data()

    with jsonlines.open(f'{args.gpt_outputs_dir}/gpt_outputs.jsonl') as f:
        generations = []
        ids =[]
        for index,line in enumerate(f):
            generation = line['generation']
            id = line['id']
            generations.append(generation)
            ids.append(id)

    assert len(ids) == len(generations)

    all_dict = all_data.all_data

    process = Process(all_dict)


    inputs,outputs = process.process_format(args.data_type,generations,ids)

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, train_size = args.ratio, random_state=42)

    splits = ['train','dev','test']
    for split in splits:
        if split == 'train':
            split_all(os.path.join(args.gpt_outputs_dir,args.data_type),X_train,y_train,split)
        else:
            split_all(os.path.join(args.gpt_outputs_dir,args.data_type),X_test,y_test,split)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split gpt output into train_dev_test')
    parser.add_argument('--all_data',default='../property_centric_process/samples_process.jsonl')
    parser.add_argument('--gpt_outputs_dir',default='./for_finetune/property_centric')
    parser.add_argument('--ratio',default=0.7,help='Ratio of the datasets for training')
    parser.add_argument('--data_type',default='wo_m_t5')
    args = parser.parse_args()
    main(args)