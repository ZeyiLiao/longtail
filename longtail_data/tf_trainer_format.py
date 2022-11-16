import os
import json
import random
import jsonlines

dir_list = ['train','test','dev']
mask = '<extra_id_0>'

random.seed(42)

def format(input,con,output,data_type):
    if 'w_m_t5' in data_type:
        input_text = f'Constraint: {con} ; Input: {input} ; Output:'
    elif 'wo_m_t5' in data_type:
        input_text = f'Constraint: {con} ; Input: {input}'
    elif 'wo_m_gpt2' in data_type:
        input_text = f'Constraint: {con} ; Input: {input}'
        output = input_text
    output_text = output
    tmp = dict()
    tmp['input'] = input_text
    tmp['output'] = output_text
    return {'translation':tmp}


def change_tf_trainer_format(root_dir,data_type):

    for dir in dir_list:
        root = f'{root_dir}/{data_type}/{dir}'
        con_path = f'{root}/constraints.jsonl'
        input_path = f'{root}/inputs.jsonl'
        output_path = f'{root}/outputs.jsonl'

        json_file = f'{root}/data.json'

        outputs = []
        inputs = []
        cons = []
        with jsonlines.open (output_path) as f:
            for line in f:
                outputs.append(line)


        with jsonlines.open (input_path) as f:

            for line in f:
                inputs.append(line)

        with jsonlines.open (con_path) as f:
            for line in f:
                cons.append(line)

        formated_json = []
        for index in range(len(inputs)):
            input = inputs[index]
            output = outputs[index]
            con = cons[index]

            formated_json.append(format(input,con,output,data_type))

        random.shuffle(formated_json)


        with open(json_file,'w') as f:
            for line in formated_json:
                json.dump(line,f)
                f.write('\n')
