import json
from operator import index
from lemminflect import getAllInflections, getInflection
import spacy
nlp = spacy.load('en_core_web_sm')
import os
import jsonlines
import csv
import copy



bad_words = ['•','do','be','-']


output_path = '/home/zeyi/longtail/property_centric_process/property_centric_samples.jsonl'

property_centric_dir = '/home/zeyi/longtail/property_centric'




with open(f'{property_centric_dir}/properties.txt') as f:
    all_properties = [line.strip() for line in f]
    

def reader_handle(reader,global_l,interval = 4000):
    for i,line in enumerate(reader):

        if i % interval == 0:
            filterd = False
            

            sent = line['base']
            if '•' in sent:
                continue
            _sent = sent[:-1]
            
            if '.' in _sent:
                continue

            sent_split = sent.split(' ')

            doc = nlp(str(sent))
            
            for index,token in enumerate(doc):
                if '[mask]' in sent_split[index]:
                    filterd = True
                    break
                if token.pos_.startswith('V'):
                    token = getInflection(str(token),tag = 'VBZ')[0]
                    sent_split[index] = str(token)
                    break
                    
            if filterd:
                continue


            verb_l = copy.deepcopy(line['constraint']['verb'])
            noun_l = copy.deepcopy(line['constraint']['noun'])
            if len(verb_l) * len(noun_l) == 0:
                continue

            checkwords = [line['object2']]
            checkwords.extend(noun_l)
            checkwords.extend(verb_l)



            if len(set(checkwords) & set(bad_words)) != 0:
                continue
            
            if len(checkwords) > 5:
                continue

            def eql_check(x,limit = 2):
                return len(x.split(' ')) <= limit

            if not all([eql_check(_) for _ in checkwords]):
                continue





            

            index_l[0] = index_l[0] + 1
            sent = ' '.join(sent_split)
            line['base'] = sent
            line['index'] = index_l[0]
            global_l.append(line)




index_l = [-1]
global_l = []
for property in all_properties:
    path = f'{property_centric_dir}/{property}'
    incre_path = f'{path}/increase_1.jsonl'
    decre_path = f'{path}/decrease_1.jsonl'


    reader = jsonlines.open(incre_path)
    reader_handle(reader,global_l)

    reader = jsonlines.open(decre_path)
    reader_handle(reader,global_l)



with jsonlines.open(output_path,'w') as f:
    f.write_all(global_l)







