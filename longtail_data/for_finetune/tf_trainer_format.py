import os
import json
import jsonlines

dir_list = ['train','test','dev']
mask = '<extra_id_0>'


def format(input,con,output):
    input_text = f'Input: {input} ; Constraint: {con} ; Output:'
    output_text = output
    tmp = dict()
    tmp['input'] = input_text
    tmp['output'] = output_text
    return {'translation':tmp}

root_dir = '/home/zeyi/longtail/longtail_data/for_finetune'
data_dir = 'w_m_t5'

for dir in dir_list:
    root = f'{root_dir}/{data_dir}/{dir}'
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

        formated_json.append(format(input,con,output))


    with open(json_file,'w') as f:
        for line in formated_json:
            json.dump(line,f)
            f.write('\n')
