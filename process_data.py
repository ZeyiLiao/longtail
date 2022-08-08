import csv
import imp
import re
from tkinter import N
import jsonlines
from collections import defaultdict as ddict,Counter
import pandas as pd

def process_gengen_tuple():
    concept_list = []
    property_list = []
    relation_list = []
    with open('./gengen.csv') as f:
        reader = csv.reader(f)
        for item in reader:
            processed_sentence = item[-4]
            concept,property,relation = item[-3],item[-2],item[-1]
            concept_list.append(concept)
            relation_list.append(relation)
            if property == '':
                if concept == "Porpoise":
                    print('2')
                    pass
                processed_sentence = processed_sentence.replace(f'{concept} ',"")
                processed_sentence = processed_sentence.replace(f'{relation} ',"")
                processed_sentence = processed_sentence.replace(" .","")
                processed_sentence = processed_sentence.replace('is ',"")
                processed_sentence = processed_sentence.replace("are ","")
                processed_sentence = processed_sentence.replace("a ","")
                processed_sentence = processed_sentence.replace("an ","")
                processed_sentence = processed_sentence.replace("to ","")

                property = processed_sentence
            property_list.append(property)

    combined = list(zip(concept_list,property_list,relation_list))
    with open('./gengen_tuple.csv','w') as f:
        writer = csv.writer(f)
        writer.writerows(combined)

def combine_atmoic_gengen():
    concept_list = []
    property_list = []
    relation_list = []
    count = 0
    with open('./data/ATOMIC10X.json','r') as file:
        reader = jsonlines.Reader(file)
        for item in reader:
            concept_list.append(item['head'])
            relation_list.append(item['relation'])
            property_list.append(item['tail'])


    with open('./data/gengen.csv') as f:
        reader = csv.reader(f)
        for item in reader:
            sen = item[1].strip(' .')
            concept_list.append(sen)
            relation_list.append('NA')
            property_list.append('NA')


    combined = list(zip(concept_list,relation_list,property_list))
    with open('./data/all_tuples.csv','w') as f:
        writer = csv.writer(f)
        writer.writerows(combined)

def extract_all_head():
    heads = []
    with open('./data/all_tuples.csv') as f:
        reader = csv.reader(f)
        for item in reader:
            heads.append(item[0])
    n = f'\n'
    with open('./data/all_heads','w') as f:
        for head in heads:
            f.write(head)
            f.write(n)

def process_atomic_tuple():
    concept_list = []
    property_list = []
    relation_list = []
    with open('./ATOMIC10X.json','r') as file:
        reader = jsonlines.Reader(file)
        for item in reader:
            concept_list.append(item['head'])
            relation_list.append(item['relation'])
            property_list.append(item['tail'])



    combined = list(zip(concept_list,property_list,relation_list))
    with open('./atomic_tuple.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['concept','property','relation'])
        writer.writerows(combined)


def find_pattern_atomic():
    text = ddict(lambda : ddict(list))
    count = 0
    df = pd.read_csv('./atomic_tuple.csv')
    for index in range(df.shape[0]):
        relation = df.loc[index,'relation']
        concept = df.loc[index,'concept']
        property = df.loc[index,'property']
        if relation == 'xIntent':
            if property[:2] != 'to':
                print(f'{concept} {relation} {property}')
                count += 1
                if count >= 20:
                    break
        # if 'PersonY' in concept and relation == 'xWant':
        #     if property[:2] != 'to':

        #         print(f'{concept} {relation} {property}')
        #         count += 1
        #         if count >= 50:
        #             break

        # if 'PersonY' in concept and relation == 'xEffect':
        # if 'PersonY' in concept and relation == 'xNeed':

        # if 'PersonY' in concept and 'PersonX' in property and 'PersonY' in property:
        # if 'PersonY' in concept and 'PersonX' not in property and 'PersonY' in property:
        # if 'PersonY' in concept and 'PersonX' in property and 'PersonY' not in property:
        #     if relation == 'xIntent':
        #         print(f'{concept} {relation} {property}')
        #         count += 1
        #         if count >= 50:
        #             break


    #     if relation == 'xAttr':
    #         property = 'PersonX is seen as ' + property
    #     elif relation == 'xReact':
    #         property = 'PersonX feels ' + property
    #     elif relation == 'xNeed':
    #         property = 'PersonX has ' + property
    #     elif relation == 'xWant':
    #         property = 'PersonX wants ' + property
    #     elif relation == 'xIntent':
    #         property = 'PersonX intents ' + property
    #     elif relation == 'xEffect':
    #         property = 'PersonX ' + property
    #     elif relation == 'HinderedBy':
    #         property = property

    #     text[relation]['concept'].append(concept)
    #     text[relation]['relation'].append(relation)
    #     text[relation]['property'].append(property)

    #     count += 1


    # for relation in text.keys():
    #     file_name = f"./atomic_tuple_{relation}.csv"
    #     df = pd.DataFrame(text[relation])
    #     df.to_csv(file_name,index=False)


def pos_neg_symbol_gengen():
    relation_map = ddict(list)
    relation_map_count = dict()
    df = pd.read_csv('./gengen_tuple.csv')
    for index in range(df.shape[0]):
        relation = df.loc[index,'relation']
        relation_map[relation].append(index)
        relation_map_count[relation] = len(relation_map[relation])
    print(Counter(relation_map_count))

combine_atmoic_gengen()
extract_all_head()