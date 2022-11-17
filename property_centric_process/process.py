
import jsonlines
import copy
from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV
import pickle

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
all_data = {}


with jsonlines.open('./samples.jsonl') as f:
    for line in f:
        ori_sent = line['base']
        index = line['index']

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

                
                _id = str(index) + f'_{conj_word}' if neg_word == '' else str(index) + f'_{conj_word}_neg'
                _line['id'] = _id
                _line['conti_template'] = conti_template
                _line['normal_template'] = normal_template
                
            
                constraints_lemma = [verb_l]
                noun_l_cnf = logic_wrapper.run(noun_l)
                constraints_lemma.extend(noun_l_cnf)

                if neg_word == 'no':
                    constraints_lemma.append([neg_word])
                _line['cons_lemma'] = constraints_lemma

                
                
                constraints_inflection = []
                for clause in constraints_lemma:
                    tmp = []
                    if clause[0] == 'no':
                        continue
                    for word in clause:

                        word_inflections = getAllInflections(word)
                        if not word_inflections or len(word_inflections) == 0:
                            word_inflections = dict(getAllInflectionsOOV(word,'VERB'), **getAllInflectionsOOV(word,'NOUN'))
                            if len(word.split(' ')) == 1:
                                word_inflections.update(getAllInflectionsOOV(word,'ADJ'))
                        tmp.extend(list(set([_[0] for _ in list(word_inflections.values())])))
                    constraints_inflection.append(tmp)

                if neg_word == 'no':
                    constraints_inflection.append(['no', 'not'])

                _line['cons_inflection'] = constraints_inflection

                all_data[_id] = _line
                jsonl_o.append(_line)



with jsonlines.open('./all_data.jsonl','w') as f:
    f.write_all(jsonl_o)



with open ('./all_data.pkl','wb') as f:
    pickle.dump(all_data,f)


        
