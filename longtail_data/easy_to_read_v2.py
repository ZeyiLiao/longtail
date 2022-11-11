import csv
import imp
import jsonlines
import json
import copy
import sacrebleu

def back_conti_sent(sent, generation, mask = '<extra_id_0>'):
    return sent.replace(mask,generation)

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
    cons = copy.deepcopy(data['constraint'])
    cons['noun'].append(object2)
    
    if has_neg:
        cons['neg'] = ['no']
    return cons
    

neuro_dict = {}

vanilla_dict = {}

gpt_dict = {}

lemma_l = []


need_index_1 = [2, 30, 34, 37, 39, 40]
need_index_2 = [0, 9, 15, 19, 23, 31, 33, 36]

need_index = list(set(need_index_1) | set(need_index_2))




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



with open('/home/zeyi/longtail/longtail_data/generated_data/property_centric/gpt_outputs.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        generation_part, id = line[0],line[1]
        gpt_dict[id] = generation_part


with open('/home/zeyi/longtail/longtail_data/raw_data/property_centric/lemma_constraints_t5_infer.json') as f:
    for i,line in enumerate(f):
        if i >= len(gpt_dict.keys()):
            break
        lemma = json.loads(line)
        lemma_l.append(lemma)
        

conti_format_dict = {}

with open('/home/zeyi/longtail/longtail_data/raw_data/property_centric/inputs_t5_infer.csv') as f:
    reader = csv.reader(f)
    for i,line in enumerate(reader):
        if i >= len(gpt_dict.keys()):
            break
        conti_format_dict[line[1]] = line[0]



neuro_o = []
vanilla_o = []

o_path = '/home/zeyi/longtail/longtail_data/generated_data/property_centric/compare.txt'
o_path_jsonl = '/home/zeyi/longtail/longtail_data/generated_data/property_centric/compare.jsonl'


fo = open(o_path,'w')
fo_jsonl = jsonlines.open(o_path_jsonl,'w')


nl = '\n'
exact_match = 100
exact_match_count = 0


for index,id in enumerate(gpt_dict.keys()):
    jsonl_dict = {}

    id_number = int(id.split('_')[0])
    conj_word = id.split('_')[1]

    has_neg = False
    if len(id.split('_')) == 3:
        has_neg = True

    if id_number not in need_index:
        continue

    ori_data = all_data[id_number]

    generation_neuro = neuro_dict[id]
    generation_vanilla = vanilla_dict[id]
    generation_gpt = gpt_dict[id]

    # cons = constraints(ori_data,has_neg)
    cons = lemma_l[index]
    length_cons = len(cons) if cons[-1][0] != 'no' else len(cons)-1

    sample_conti = ori_data['sample_cont']
    base = ori_data['base']
    jsonl_dict['Base'] = base
    jsonl_dict['id'] = id
    jsonl_dict['Constraints'] = cons
    jsonl_dict['Constraints_length'] = length_cons
    jsonl_dict['Sample_continuation'] = sample_conti
    fo.write(f'Base: {base} ,  id :{id}' )
    fo.write(nl)
    fo.write(f'Constraints: {cons}')
    fo.write(nl)
    fo.write(f'Sample continuation: {sample_conti}')
    fo.write(nl)
    fo.write(nl)
    fo.write('Training and inference format')
    fo.write(nl)
    fo.write(f'Input : {conti_format_dict[id]} ; Constrants : {cons} ; Output : ')
    fo.write(nl)

    fo.write(nl)
    neuro = back_conti_sent(conti_format_dict[id],generation_neuro)
    vanilla = back_conti_sent(conti_format_dict[id],generation_vanilla)
    fo.write(f'Neuro: {neuro}')
    fo.write(nl)
    fo.write(f'Vanilla: {vanilla}')
    fo.write(nl)
    bleu_score = sacrebleu.corpus_bleu([neuro], [[vanilla]]).score
    if bleu_score >= exact_match:
        exact_match_count += 1
    fo.write(f'bleu: {bleu_score}')

    fo.write(nl)
    fo.write(f'GPT-3: {back_conti_sent(conti_format_dict[id],generation_gpt)}')

    jsonl_dict['neruo'] = neuro
    jsonl_dict['vanilla'] = vanilla
    jsonl_dict['GPT3'] = back_conti_sent(base,generation_gpt)

    fo_jsonl.write(jsonl_dict)

    fo.write(nl)
    fo.write(nl)
    fo.write(nl)
    fo.write('*******************************')
    fo.write(nl)
    fo.write(nl)
    fo.write(nl)
    
    vanilla_o.append(back_conti_sent(base,generation_vanilla))
    neuro_o.append(back_conti_sent(base,generation_neuro))


fo.write(nl)
fo.write(f'Ratio of exach match: {exact_match_count/len(gpt_dict.keys())}')

fo.write(nl)
fo.close()






