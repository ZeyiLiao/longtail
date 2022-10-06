import numpy as np
from pathlib import Path

import jsonlines

with open('./GPT_fill_output.txt') as f:
    inputs = []
    constraints = []
    outputs = []
    bad_outputs_index = []

    group_index = -1
    can_start = False
    next_output = False
    for line in f:
        line = line.replace('\n','')
        if line == '' or can_start:
            can_start = True
        else:
             continue

        if line.startswith('input:'):
            group_index += 1
            inputs.append(line[7:].strip().replace('[mask]','<extra_id_0>'))
        elif line.startswith('constraint:'):
            constraints.append(line[12:])
        elif line.startswith('output:'):
            next_output = True
        elif next_output:
            outputs.append(line.strip())

            if line == '' or line == 'None':
                bad_outputs_index.append(group_index)

            next_output = False

bad_outputs_index.sort(reverse=True)

for _ in bad_outputs_index:
    del inputs[_]
    del constraints[_]
    del outputs[_]

assert '' not in outputs
assert len(inputs) == len(constraints) == len(outputs)

train_count = int(len(inputs)*0.75)
train_index = np.random.choice(len(inputs),train_count,replace=False)

train_inputs = []
train_constraints = []
train_outputs = []
test_inputs = []
test_constraints = []
test_outputs = []


for index in range(len(inputs)):
    if index in train_index:
        train_inputs.append(inputs[index])
        train_constraints.append(constraints[index])
        train_outputs.append(outputs[index])

    else:
        test_inputs.append(inputs[index])
        test_constraints.append(constraints[index])
        test_outputs.append(outputs[index])

root_dir = '/home/zeyi/finetune/Data/raw'
train_dir = root_dir + '/train'
test_dir = root_dir + '/test'
dev_dir = root_dir + '/dev'

Path(f'{train_dir}').mkdir(exist_ok=True,parents=True)
Path(f'{test_dir}').mkdir(exist_ok=True,parents=True)
Path(f'{dev_dir}').mkdir(exist_ok=True,parents=True)


with jsonlines.open(f'{train_dir}/inputs.jsonl', mode='w') as writer:
    writer.write_all(train_inputs)
with jsonlines.open(f'{train_dir}/constraints.jsonl', mode='w') as writer:
    writer.write_all(train_constraints)
with jsonlines.open(f'{train_dir}/outputs.jsonl', mode='w') as writer:
    writer.write_all(train_outputs)


with jsonlines.open(f'{test_dir}/inputs.jsonl', mode='w') as writer:
    writer.write_all(test_inputs)
with jsonlines.open(f'{test_dir}/constraints.jsonl', mode='w') as writer:
    writer.write_all(test_constraints)
with jsonlines.open(f'{test_dir}/outputs.jsonl', mode='w') as writer:
    writer.write_all(test_outputs)


with jsonlines.open(f'{dev_dir}/inputs.jsonl', mode='w') as writer:
    writer.write_all(test_inputs)
with jsonlines.open(f'{dev_dir}/constraints.jsonl', mode='w') as writer:
    writer.write_all(test_constraints)
with jsonlines.open(f'{dev_dir}/outputs.jsonl', mode='w') as writer:
    writer.write_all(test_outputs)
