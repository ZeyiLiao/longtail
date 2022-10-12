from curses import nl
import json
from random import random
import numpy as np

from pathlib import Path



from process_t5_format import process_t5_format
import jsonlines
import argparse
from tf_trainer_format import change_tf_trainer_format


np.random.seed(42)

def main(args):
    root_dir = args.gpt_outputs_dir

    with jsonlines.jsonlines.open(f'{root_dir}/gpt_outputs.jsonl') as f:
        inputs = []
        constraints = []
        outputs = []
        bad_outputs_index = []
        for index,line in enumerate(f):

            inputs.append(line['input'])
            constraints.append(line['constraint'])
            output = line['generation']
            outputs.append(output)

            if output == '' or output == 'None':
                bad_outputs_index.append(index)


    bad_outputs_index.sort(reverse=True)

    for _ in bad_outputs_index:
        del inputs[_]
        del constraints[_]
        del outputs[_]

    assert '' not in outputs
    assert len(inputs) == len(constraints) == len(outputs)

    inputs,outputs = process_t5_format(inputs,outputs)


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

    data_dir = 'w_m_t5'
    root_data_dir = f'{root_dir}/{data_dir}'
    train_dir = root_data_dir + '/train'
    test_dir = root_data_dir + '/test'
    dev_dir = root_data_dir + '/dev'

    Path(f'{train_dir}').mkdir(exist_ok=True,parents=True)
    Path(f'{test_dir}').mkdir(exist_ok=True,parents=True)
    Path(f'{dev_dir}').mkdir(exist_ok=True,parents=True)

    nl = '\n'
    with jsonlines.open(f'{train_dir}/inputs.jsonl', mode='w') as f:
        f.write_all(train_inputs)

    with jsonlines.open(f'{train_dir}/constraints.jsonl', mode='w') as f:
        f.write_all(train_constraints)

    with jsonlines.open(f'{train_dir}/outputs.jsonl', mode='w') as f:
        f.write_all(train_outputs)


    with jsonlines.open(f'{test_dir}/inputs.jsonl', mode='w') as f:
        f.write_all(test_inputs)

    with jsonlines.open(f'{test_dir}/constraints.jsonl', mode='w') as f:
        f.write_all(test_constraints)

    with jsonlines.open(f'{test_dir}/outputs.jsonl', mode='w') as f:
        f.write_all(test_outputs)



    with jsonlines.open(f'{dev_dir}/inputs.jsonl', mode='w') as f:
        f.write_all(test_inputs)

    with jsonlines.open(f'{dev_dir}/constraints.jsonl', mode='w') as f:
        f.write_all(test_constraints)

    with jsonlines.open(f'{dev_dir}/outputs.jsonl', mode='w') as f:
        f.write_all(test_outputs)

    change_tf_trainer_format(root_dir,data_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split gpt output into train_dev_test')
    parser.add_argument('--gpt_outputs_dir')
    args = parser.parse_args()
    main(args)