import json
import random
import argparse
import csv

def main(args):
    inputs_dir = args.inputs_dir
    datas_dir = args.datas_dir

    inputs = []
    gpt_outputs = []
    neuro_outputs = []
    vanilla_outputs = []
    lemmatized_cons = []
    infection_cons = []

    mask = '<extra_id_0>'

    def change_format(x):
        neg = []
        if len(x) == 2:
            neg = x[1]
        cons = x[0]
        tmp = []
        for _ in cons:
            tmp.append(_)
        random.shuffle(tmp)

        cons_string = '[' + ', '.join(tmp) + ']'
        if len(neg) != 0:
            cons_string = cons_string + f', [{neg[0]}]'

        return cons_string


    with open(f'{inputs_dir}/inputs_t5_infer.csv') as f:
        inputs = []
        order_ori = []
        reader = csv.reader(f)
        for line in reader:
            x,order = line[0],line[1]
            inputs.append(x.replace('\n',''))
            order_ori.append(order)


    with open(f'{inputs_dir}/lemma_constraints_t5_infer.json') as f:
        lemmatized_cons = []
        for line in f:
            x = change_format(json.loads(line))
            lemmatized_cons.append(x)

    with open(f'{inputs_dir}/inflection_constraints_t5_infer.json') as f:
        infection_cons = []
        for line in f:
            x = json.loads(line)
            infection_cons.append(x)


    with open(f'{datas_dir}/gpt_outputs.csv') as f:
        gpt_outputs = []
        order_gpt = []
        reader = csv.reader(f)
        for index,line in enumerate(reader):
            x,order = line[0],line[1]
            if 'but' in order:
                x = x[0].upper() + x[1:]
            gpt_outputs.append(inputs[index].replace(mask,x.replace('\n','')))
            order_gpt.append(order)


    with open(f'{datas_dir}/t5_3b_w_m.csv') as f:
        neuro_outputs = []
        order_neuro = []
        reader = csv.reader(f)
        for index,line in enumerate(reader):
            x,order = line[0],line[1]
            if 'but' in order:
                x = x[0].upper() + x[1:]
            neuro_outputs.append(inputs[index].replace(mask,x.replace('\n','')))
            order_neuro.append(order)



    with open(f'{datas_dir}/t5_3b_vanilla_w_m.csv') as f:
        vanilla_outputs = []
        order_vanilla = []
        reader = csv.reader(f)
        for index,line in enumerate(reader):
            x,order = line[0],line[1]
            if 'but' in order:
                x = x[0].upper() + x[1:]
            vanilla_outputs.append(inputs[index].replace(mask,x.replace('\n','')))
            order_vanilla.append(order)




    if args.num_groups != -1:
        if args.variations_per_group == -1:
            return NotImplementedError
        else:
            group_count = args.variations_per_group

        all_indexs = []
        for group in range(args.num_groups):
            indexs = range(group*group_count,group*group_count+group_count)
            # selected_indexs = sorted(random.sample(indexs,2))
            selected_indexs = indexs
            all_indexs.extend(list(selected_indexs))

        inputs = [inputs[i] for i in all_indexs]
        gpt_outputs = [gpt_outputs[i] for i in all_indexs]
        neuro_outputs = [neuro_outputs[i] for i in all_indexs]
        vanilla_outputs = [vanilla_outputs[i] for i in all_indexs]
        lemmatized_cons = [lemmatized_cons[i] for i in all_indexs]
        infection_cons = [infection_cons[i] for i in all_indexs]


    assert len(inputs) == len(gpt_outputs) == len(neuro_outputs) == len(vanilla_outputs) == len(lemmatized_cons) == len(infection_cons)



    nl = '\n'
    f = open(f'{datas_dir}/comparison_file.txt','w')
    f_json = open(f'{datas_dir}/comparison_file.json','w')

    for index in range(len(inputs)):
        f.write(f'input_format:')
        f.write(nl)
        f.write(f'Input: {inputs[index]} ; Constraint: {str(lemmatized_cons[index])} ; Output:')
        f.write(nl)
        f.write(nl)
        f.write(f'These are constraints inflections used only for neuro algorithm')
        f.write(nl)
        f.write(str(infection_cons[index]))
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='easy to rad')
    parser.add_argument('--inputs_dir',default = '../longtail_data/raw_data')
    parser.add_argument('--datas_dir',default = '.')

    parser.add_argument('--num_groups', default= -1, type=int)
    parser.add_argument('--variations_per_group', default= -1, type=int)
    args = parser.parse_args()
    main(args)
