import math
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import argparse
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_fig(root_path,info,plot_data,title,max_bin):
    save_dir = f'{root_path}/{info}_{title}.png'
    n_bins = np.arange(1,max_bin)
    fig,ax = plt.subplots(figsize=(7,7))
    ax.hist(plot_data,n_bins)
    ax.set_title(f'{info}_{title}')
    fig.savefig(f'{save_dir}')


class GPTppl():
    def __init__(self,device):

        self.model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.device = torch.device(device)
        self.model.to(self.device)

    def calculate_ppl(self,composed_rules):
        features = [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(composed_rule))) for composed_rule in composed_rules]
        ppl_all = []

        self.model.eval()

        with torch.no_grad():
            for index,feature in enumerate(features):
                feature = feature.to(self.device)
                loss = self.model(
                    feature,
                    labels = feature,
                    return_dict = True
                ).loss
                ppl_all.append(math.exp(loss.item()))

        return ppl_all



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inputs = []
    outputs = []
    constraints = []
    model = args.model


    if 'bart' in model:
        mask = '<mask>'
    elif 't5' in model:
        mask = '<extra_id_0>'


    path = f"{args.inputs}/{args.model}_w_m.txt"
    with open(path) as f:
        inputs = [x.rstrip() for x in f.readlines()]


    path = f"{args.outputs}/{args.model}_{args.suffix}.txt"
    with open(path) as f:
        outputs = [x.rstrip() for x in f.readlines()]


    path = f"{args.constraints}/constraint_inflections.json"
    with open(path) as f:
        for line in f.readlines():
            tmp = json.loads(line)
            constraints.append(tmp)

    assert len(inputs) == len(constraints) == len(outputs)
    before_filtration = len(inputs)



    filled_part_l = []
    selected_pattern_index = []
    for index,input in enumerate(inputs):
        index_mask = input.index(mask)
        prefix_end = index_mask-1
        suffix_start = len(input) - (index_mask + len(mask)) -1
        output = outputs[index]

        constraint = constraints[index]
        filled_part = output[prefix_end+1:-suffix_start]
        filled_part_l.append(filled_part)

    #   filter those not follow the mask pattern
        if output[:prefix_end] == input[:prefix_end] and output[-suffix_start:] == input[-suffix_start:]:

            filled_words = filled_part.replace(',','').strip().split(' ')

            clause_states = []
            for concepts in constraint:

    #  filter those not follow the constraints
                clause_satisified = False
                for concept in concepts:
                    if concept in filled_words:
                        clause_satisified = True
                        break

                clause_states.append(clause_satisified)
                if clause_states[-1] == False:
                    break

            if all(clause_states):
                selected_pattern_index.append(index)






    filled_lengths = [len(filled_part_l[i].strip().split(' ')) for i in selected_pattern_index]
    inputs = [inputs[i] for i in selected_pattern_index]
    constraints = [constraints[i] for i in selected_pattern_index]
    outputs = [outputs[i] for i in selected_pattern_index]
    assert len(inputs) == len(constraints) == len(outputs) == len(filled_lengths)

    after_filtration = len(inputs)



    gpt_ppl = GPTppl(device)
    ppl_all = []

    bs = args.bs

    for batch in chunks(outputs,bs):
        ppl = gpt_ppl.calculate_ppl(batch)
        ppl_all.extend(ppl)


    Path(args.save_dir).mkdir(parents = True,exist_ok=True)



    plot_fig(f'{args.save_dir}', info = f'{args.model}_{args.suffix}',plot_data = ppl_all,title = 'Perplexity',max_bin = 800)
    plot_fig(f'{args.save_dir}', info = f'{args.model}_{args.suffix}',plot_data = filled_lengths, title = 'filled_length',max_bin = 13)


    nl = '\n'
    with open(f'{args.save_dir}/{args.model}_{args.suffix}.txt','w') as f:

        f.write(f'ratio of filtration is {1 -after_filtration/before_filtration}')
        f.write(nl)
        f.write('all ppl score')
        f.write(nl)
        f.write(str(ppl_all))
        f.write('all filled part length')
        f.write(nl)
        f.write(str(filled_lengths))
        for index in range(len(outputs)):
            f.write('Original sent:')
            f.write(nl)
            f.write(inputs[index])
            f.write(nl)
            f.write('constraints')
            f.write(nl)
            f.write(str(constraints[index]))
            f.write(nl)
            f.write(f'{model} output')
            f.write(nl)
            f.write(outputs[index])
            f.write(nl)
            f.write('Perplexity score: {ppl_all[index]}')
            f.write(nl)
            f.write('Length of filled part: {filled_lengths[index]}')
            f.write(nl)
            f.write('****************************')
            f.write(nl)
            f.write(nl)
            f.write(nl)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter unqualified output')
    parser.add_argument('--inputs', type=str, default= './inputs')
    parser.add_argument('--outputs', type = str, default= './outputs')
    parser.add_argument('--constraints', type = str, default= './constraints')

    parser.add_argument('--model',  choices=['t5_3b','t5_large'])
    parser.add_argument('--suffix', type = str, help='it can be w_m or wo_m or w_m_beam')
    parser.add_argument('--save_dir', type = str, default='./filtered_generation')
    parser.add_argument('--bs',  default=256 , type = int)

    main(parser.parse_args())
