
import jsonlines
import copy
from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV
import random
random.seed(37)
from split_train_infer import split_train_infer



def change_format(sent, conj_word , if_conti = True, mask = '[mask]'):
    mask_index = sent.index(mask)
    head = sent[:mask_index-1]
    assert 'If' in head,'If is not in head'
    
    
    head = head[3:]
    
    tail = sent[mask_index + len(mask) + 2:-1]

    tail = tail[0].upper() + tail[1:]
    head = head[0].lower() + head[1:]


    if if_conti:

        if conj_word == 'and':
            sent = tail + ' because ' + head + ' and ' + mask + '.'
        elif conj_word == 'while':
            sent = tail + ' because ' + head + ' while ' + mask + '.'

        elif conj_word == 'but':
            sent = tail + ' because ' + head + ' despite the fact that ' + mask + '.'

        elif conj_word == 'although':
            sent = tail + ' because ' + head + ' even though ' + mask + '.'

        return sent.replace(mask,'<extra_id_0>')


    else:
        raise NotImplementedError


conj_words = ['and', 'while', 'but', 'although']
negation_word = ['no']




inputs_o = []
indexs_o = []
cons_lemma_o = []
cons_inflec_o = []
o_path = '/home/zeyi/longtail/longtail_data'



with jsonlines.open('/home/zeyi/longtail/property_centric_process/property_centric_samples.jsonl') as f:
    for line in f:
        ori_sent = line['base']

        
        for conj_word in conj_words:
            formatted_inputs = []
            formatted_inputs.append(change_format(ori_sent,conj_word))
            # negation, so we times 2
            formatted_inputs = formatted_inputs * 2 
            inputs_o.extend(formatted_inputs)



        index = line['index']
        for conj_word in conj_words:
            _index = str(index) + f'_{conj_word}'
            indexs_o.append(_index)

            _index = _index + f'_neg'
            indexs_o.append(_index)




        verb_l = copy.deepcopy(line['constraint']['verb'])
        noun_l = copy.deepcopy(line['constraint']['noun'])
        noun_l.append(line['object2'])



        for _ in range(len(conj_words)):
            constraints_lemma = [verb_l,noun_l]
            cons_lemma_o.append(constraints_lemma)

            
            tmp = copy.deepcopy(constraints_lemma)
            tmp.append(['no'])
            cons_lemma_o.append(tmp)



        constraints_inflection = []
        for clause in constraints_lemma:
            tmp = []
            for word in clause:

                word_inflections = getAllInflections(word)
                if not word_inflections or len(word_inflections) == 0:
                    word_inflections = dict(getAllInflectionsOOV(word,'VERB'), **getAllInflectionsOOV(word,'NOUN'))
                tmp.extend(list(set([_[0] for _ in list(word_inflections.values())])))
            constraints_inflection.append(tmp)



        for _ in range(len(conj_words)):
            cons_inflec_o.append(constraints_inflection)
            tmp = copy.deepcopy(constraints_inflection)
            tmp.append(['no', 'not'])
            
            cons_inflec_o.append(tmp)



assert len(inputs_o)== len(indexs_o)== len(cons_lemma_o)== len(cons_inflec_o)
inputs_indexs_o = list(zip(inputs_o,indexs_o))

variation_per_sent = len(conj_words) * (len(negation_word) + 1)
groups_for_train = 30

groups_for_train = random.sample(range( int(len(inputs_o)/variation_per_sent) ) ,groups_for_train)

split_args = dict()
split_args['fi_write'] = inputs_indexs_o
split_args['fc_lemma_write'] = cons_lemma_o
split_args['fc_inflect_write'] = cons_inflec_o
split_args['num_variations'] = variation_per_sent
split_args['groups_for_train'] = groups_for_train
split_args['output_file'] = o_path
split_args['model_name'] = 't5'

split_train_infer(split_args,sub_folder = 'property_centric')



        
