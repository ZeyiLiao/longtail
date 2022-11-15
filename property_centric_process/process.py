
import jsonlines
import copy

import random
random.seed(37)

from utils import Logic_wrapper


logic_wrapper = Logic_wrapper()

def change_format(sent, conj_word , if_conti = True, mask = '[mask]'):
    mask_index = sent.index(mask)
    head = sent[:mask_index].strip()
    assert 'If' in head,'If is not in head'
    
    
    head = head.replace('If','').strip()
    
    tail = sent[mask_index + len(mask) + 2:-1]




    if if_conti:
        tail = tail[0].upper() + tail[1:]
        head = head[0].lower() + head[1:]

        if conj_word == 'and':
            sent = tail + ' because ' + head + ' and ' + mask + '.'
        elif conj_word == 'while':
            sent = tail + ' because ' + head + ' while ' + mask + '.'

        elif conj_word == 'but':
            sent = tail + ' because ' + head + ' despite the fact that ' + mask + '.'

        elif conj_word == 'although':
            sent = tail + ' because ' + head + ' even though ' + mask + '.'

        return sent


    else:
        if conj_word == 'and':
            sent = head + ' and ' + mask + ', so ' + tail + '.'
        elif conj_word == 'while':
            sent = head + ' while ' + mask + ', so ' + tail + '.'
        elif conj_word == 'but':
            sent = mask  + ' but ' + head + ', so ' + tail + '.'
        elif conj_word == 'although':
            sent = 'Although ' + mask  + ', '+ head + ', so ' + tail + '.'
        sent = sent.capitalize()
        
        return sent


conj_words = ['and', 'while', 'but', 'although']
conj_words = ['and']
neg_words = ['','no']


jsonl_o = []

inputs_o = []
indexs_o = []
cons_lemma_o = []
cons_inflec_o = []
o_path = '/home/zeyi/longtail/longtail_data'



with jsonlines.open('./samples.jsonl') as f:
    for line in f:
        ori_sent = line['base']
        id = line['index']

        verb_l = copy.deepcopy(line['constraint']['verb'])
        noun_l = copy.deepcopy(line['constraint']['noun'])
        noun_l.append(line['object2'])

        verb_l = list(set(verb_l))
        noun_l = list(set(noun_l))
        line.pop('constraint')
        line['verb'] = verb_l
        line['noun'] = noun_l

        for conj_word in conj_words:
            for neg_word in neg_words:
                _line = copy.deepcopy(line)
            
                conti_template = change_format(ori_sent,conj_word)
                normal_template = change_format(ori_sent,conj_word,if_conti=False)

                # negation, so we times have two time
                
                _id = str(id) + f'_{conj_word}' if neg_word == '' else str(id) + f'_{conj_word}_neg'
                _line['id'] = _id
                _line['conti_template'] = conti_template
                _line['normal_template'] = normal_template
                
            
                constraints_lemma = [verb_l]
                noun_l_cnf = logic_wrapper.run(noun_l)
                constraints_lemma.extend(noun_l_cnf)

                if neg_word == 'no':
                    constraints_lemma.append([neg_word])
                _line['cons_lemma'] = constraints_lemma
                
                jsonl_o.append(_line)

with jsonlines.open('./samples_process.jsonl','w') as f:
    f.write_all(jsonl_o)




        
