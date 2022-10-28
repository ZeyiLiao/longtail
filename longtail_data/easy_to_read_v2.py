import csv
import imp
import jsonlines
import json


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
    cons = data['constraint']
    cons['object2'] = object2
    if has_neg:
        cons['neg'] = 'no'
    return cons
    

neuro_dict = {}

vanilla_dict = {}


with open('/home/zeyi/longtail/property_centric_process/property_centric_samples.jsonl') as f:
    all_data = [json.loads(line) for line in f.readlines()]


with open('/home/zeyi/longtail/longtail_data/generated_data/property_centric/t5_3b_w_m.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        generation_part, id = line[0],line[1]

        neuro_dict[id] = generation_part


with open('/home/zeyi/longtail/longtail_data/generated_data/property_centric/t5_3b_vanilla_w_m.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        generation_part, id = line[0],line[1]
        vanilla_dict[id] = generation_part




neuro_o = []
vanilla_o = []

o_path = '/home/zeyi/longtail/longtail_data/generated_data/property_centric/compare.txt'

fo = open(o_path,'w')
nl = '\n'

for id in neuro_dict.keys():
    id_number = int(id.split('_')[0])
    conj_word = id.split('_')[1]

    has_neg = False
    if len(id.split('_')) == 3:
        has_neg = True

    ori_data = all_data[id_number]

    generation_neuro = neuro_dict[id]
    generation_vanilla = vanilla_dict[id]

    cons = constraints(ori_data,has_neg)
    sample_conti = ori_data['sample_cont']
    base = ori_data['base']

    fo.write(f'Base: {base}')
    fo.write(nl)
    fo.write(f'Constraints: {cons}')
    fo.write(nl)
    fo.write(f'Sample continuation: {sample_conti}')
    fo.write(nl)
    fo.write(nl)
    fo.write(f'Neruo: {back_sent(base,conj_word,generation_neuro)}')
    fo.write(nl)
    fo.write(f'Vanilla: {back_sent(base,conj_word,generation_vanilla)}')
    fo.write(nl)
    fo.write(nl)
    fo.write(nl)
    fo.write('*******************************')
    fo.write(nl)
    fo.write(nl)
    fo.write(nl)
    
    vanilla_o.append(back_sent(base,conj_word,generation_vanilla))
    neuro_o.append(back_sent(base,conj_word,generation_neuro))

fo.close()
