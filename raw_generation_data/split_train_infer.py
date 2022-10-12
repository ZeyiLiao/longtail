import json
from pathlib import Path
import csv

def split_train_infer(split_args,for_dis = False):

    fi_write = split_args['fi_write']
    fc_lemma_write = split_args['fc_lemma_write']
    fc_inflect_write = split_args['fc_inflect_write']
    num_variations = split_args['num_variations']
    groups_for_train = split_args['groups_for_train']
    model_name = split_args['model_name']
    output_file = split_args['output_file']

    train_inputs = []
    infer_inputs = []
    train_lemmas = []
    infer_lemmas = []
    train_inflections = []
    infer_inflections = []


    for index in range(0,len(fc_inflect_write),num_variations):
        if index//num_variations in groups_for_train:
            train_inputs.extend(fi_write[index:index+num_variations])
            train_lemmas.extend(fc_lemma_write[index:index+num_variations])
            train_inflections.extend(fc_inflect_write[index:index+num_variations])
        else:
            infer_inputs.extend(fi_write[index:index+num_variations])
            infer_lemmas.extend(fc_lemma_write[index:index+num_variations])
            infer_inflections.extend(fc_inflect_write[index:index+num_variations])


    root_dir = f'{output_file}/raw_data'
    if for_dis:
        root_dir += '/for_dis'

    Path(root_dir).mkdir(parents= True,exist_ok=True)
    root_dir = Path(root_dir)


    train_or_infer = 'train'
    nl = '\n'

    generation_inputs_path = root_dir / f'inputs_{model_name}_{train_or_infer}.csv'
    generation_constraints_lemmas_path = root_dir / f'lemma_constraints_{model_name}_{train_or_infer}.json'
    generation_constraints_inflections_path = root_dir / f'inflection_constraints_{model_name}_{train_or_infer}.json'

    fi = open(generation_inputs_path,'w')
    fi_w = csv.writer(fi)
    fi_w.writerows(train_inputs)

    fc_lemma = open(generation_constraints_lemmas_path,'w')
    fc_inflect = open(generation_constraints_inflections_path,'w')

    for index in range(len(train_inputs)):

        json.dump(train_lemmas[index],fc_lemma)
        fc_lemma.write(nl)
        json.dump(train_inflections[index],fc_inflect)
        fc_inflect.write(nl)

    fi.close()
    fc_lemma.close()
    fc_inflect.close()

# **************************************************************************
    train_or_infer = 'infer'
    nl = '\n'

    generation_inputs_path = root_dir / f'inputs_{model_name}_{train_or_infer}.csv'
    generation_constraints_lemmas_path = root_dir / f'lemma_constraints_{model_name}_{train_or_infer}.json'
    generation_constraints_inflections_path = root_dir / f'inflection_constraints_{model_name}_{train_or_infer}.json'


    fi = open(generation_inputs_path,'w')
    fi_w = csv.writer(fi)
    fi_w.writerows(infer_inputs)

    fc_lemma = open(generation_constraints_lemmas_path,'w')
    fc_inflect = open(generation_constraints_inflections_path,'w')

    for index in range(len(infer_inputs)):
        json.dump(infer_lemmas[index],fc_lemma)
        fc_lemma.write(nl)
        json.dump(infer_inflections[index],fc_inflect)
        fc_inflect.write(nl)

    fi.close()
    fc_lemma.close()
    fc_inflect.close()
