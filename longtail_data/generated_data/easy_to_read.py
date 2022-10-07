import json
import random


inputs = []
gpt_outputs = []
neuro_outputs = []
vanilla_outputs = []
lemmatized_cons = []
infection_cons = []


num_groups = 20

def change_format(x):
    return x.replace('[','').replace(']','').replace("\"","").replace(', not','')

with open('../raw_data/generation_input_t5_infer.txt') as f:
    inputs = [x.replace('\n','') for x in f.readlines()]

with open('./gpt_output.txt') as f:
    gpt_outputs = [x.replace('\n','') for x in f.readlines()]

with open('./t5_3b_w_m.txt') as f:
    neuro_outputs = [x.replace('\n','') for x in f.readlines()]

with open('./t5_3b_vanilla_w_m.txt') as f:
    vanilla_outputs = [x.replace('\n','') for x in f.readlines()]

with open('../raw_data/generation_constraints_lemmas_t5_infer.json') as f:
    lemmatized_cons = [x.replace('\n','') for x in f.readlines()]
    lemmatized_cons = list(map(change_format,lemmatized_cons))

with open('../raw_data/generation_constraints_inflections_t5_infer.json') as f:
    infection_cons = [x.replace('\n','') for x in f.readlines()]

if num_groups != -1:

    all_indexs = []
    for group in range(num_groups):
        indexs = range(group*32,group*32+32)
        selected_indexs = sorted(random.sample(indexs,2))
        all_indexs.extend(list(selected_indexs))

    inputs = [inputs[i] for i in all_indexs]
    gpt_outputs = [gpt_outputs[i] for i in all_indexs]
    neuro_outputs = [neuro_outputs[i] for i in all_indexs]
    vanilla_outputs = [vanilla_outputs[i] for i in all_indexs]
    lemmatized_cons = [lemmatized_cons[i] for i in all_indexs]
    infection_cons = [infection_cons[i] for i in all_indexs]


assert len(inputs) == len(gpt_outputs) == len(neuro_outputs) == len(vanilla_outputs) == len(lemmatized_cons) == len(infection_cons)



nl = '\n'
f = open('./comparison_file.txt','w')
f_json = open('./comparison_file.json','w')

for index in range(len(inputs)):
    f.write(f'input_format:')
    f.write(nl)
    f.write(f'Input: {inputs[index]} ; Constraint: {lemmatized_cons[index]} ; Output:')
    f.write(nl)
    f.write(nl)
    f.write(f'These are constraints inflections used only for neuro algorithm')
    f.write(nl)
    f.write(infection_cons[index])
    f.write(nl)
    f.write(nl)
    f.write(nl)
    f.write(f'gpt : {gpt_outputs[index]}')
    f.write(nl)
    f.write(f'neuro : {neuro_outputs[index]}')
    f.write(nl)
    f.write(f'vanilla : {vanilla_outputs[index]}')
    f.write(nl)
    f.write(nl)
    f.write(nl)
    f.write('************************')
    f.write(nl)
    f.write(nl)
    tmp = dict()
    tmp['input'] = f'Input: {inputs[index]} ; Constraint: {lemmatized_cons[index]} ; Output:'
    tmp['cons'] = infection_cons[index]
    tmp['gpt'] = gpt_outputs[index]
    tmp['neuro'] = neuro_outputs[index]
    tmp['vanilla'] = vanilla_outputs[index]
    json.dump(tmp,f_json)
    f_json.write(nl)