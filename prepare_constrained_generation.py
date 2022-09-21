import pickle
from helper import *
from get_related_words import get_related_words
import random
import os
from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV
import nltk

import argparse
from argparse import ArgumentParser

def back_to_sen(head,relation,tail):
    if relation == 'xAttr':
        content = head + ', so PersonX is seen as ' + tail + '.'
    elif relation == 'xReact':
        content = head + ', so PersonX feels ' + tail + '.'
    elif relation == 'xNeed':
        content = 'Before ' + head + ', PersonX has ' + tail + '.'
    elif relation == 'xWant':
        content = head + ', so PersonX wants ' + tail + '.'
    elif relation == 'xIntent':
        content = head + ', because PersonX intents ' + tail + '.'
    elif relation == 'xEffect':
        content = head + ', as a result, PersonX ' + tail + '.'
    elif relation == 'HinderedBy':
        content =  head + ', this is hindered if ' + tail + '.'
    return content


def back_to_sen_mask(head,relation,tail,model_name):
    if 'bart' in model_name:
        mask = ' <mask> '
    elif 't5' in model_name:
        mask = ' <extra_id_0> '

    if relation == 'xAttr':
        content = head + mask + 'so PersonX is seen as ' + tail + '.'
    elif relation == 'xReact':
        content = head + mask + 'so PersonX feels ' + tail + '.'
    return content


def main(args):

    model_name = args.model_name

    premise_inputs = dict()
    premise_constraints = dict()

    generation_inputs_path = f'./generation_input_{model_name}.txt'
    generation_constraints_path = f'./generation_constraints_{model_name}.json'
    generation_constraints_inflections_path = f'./generation_constraints_inflections_{model_name}.json'

    if os.path.exists(generation_constraints_path):
        os.remove(generation_constraints_path)

    if os.path.exists(generation_inputs_path):
        os.remove(generation_inputs_path)

    if os.path.exists(generation_constraints_inflections_path):
        os.remove(generation_constraints_inflections_path)



    if not os.path.exists(args.save_file) or args.force == True:

        get_related_words(args.input_file,args.save_file)


    with open(args.save_file,'rb') as f:
        premise_words = pickle.load(f)

    with open('extracted_words.pkl','rb') as f:
        premise_extractions = pickle.load(f)

    with open('./related_words.txt','w+') as f:
        for premise in premise_words.keys():
            f.write('Original premise:')
            f.write(premise)
            f.write('\n')
            f.write('\n')
            f.write('extracted words')
            f.write(str(premise_extractions[premise]))
            f.write('\n')
            f.write('\n')
            f.write('related words')
            f.write(str(premise_words[premise]))
            f.write('\n')
            f.write('\n')
            f.write('***********************')
            f.write('\n')
            f.write('\n')



    all_tuples_file = args.csv_file

    df = pd.read_csv(all_tuples_file,names = ['head','relation','tail'],index_col='head')

    for query in premise_words.keys():
        df_selected = df.loc[query]
        relations = list(df_selected['relation'])
        tails = list(df_selected['tail'])
        attr_react_set = {'xReact','xAttr'}
        desired_relations = list(set(relations) & attr_react_set)


        if len(desired_relations) == 0:
            continue

        relations_tails = []
        count = 0
        for index,relation in enumerate(relations):
            if relation in desired_relations:
                relations_tails.append((relation,tails[index]))
                count += 1;
                desired_relations.remove(relation)

            if len(desired_relations) == 0:
                break

            if count == 2:
                break


        backed_sents = []
        for index,relation_tail in enumerate(relations_tails):
            relation,tail = relation_tail[0],relation_tail[1]
            backed_sent = back_to_sen_mask(query,relation,tail,model_name)
            backed_sents.append(backed_sent)

        premise_inputs[query] = backed_sents



        words_candidates = premise_words[query]

        sample_times = len(words_candidates)

        words_candidates = random.sample(words_candidates,sample_times)

        premise_constraints[query] = words_candidates






    conjunction_words = ['and']

    negation_words = ['no']



    fi = open(generation_inputs_path,'a+')
    fc = open(generation_constraints_path,'a+')
    fc_inflect = open(generation_constraints_inflections_path,'a+')

    for premise in premise_constraints.keys():


        # 一个句子有多少个变种
        factor = len(premise_constraints[premise]) * len(conjunction_words) * (len(negation_words) + 1)

        # premise_inputs
        for backed_sent in premise_inputs[premise]:
            for _ in range(factor):
                fi.write(backed_sent)
                fi.write('\n')




        # premise_inputs
        for _ in range(len(premise_inputs[premise])):

            for word in premise_constraints[premise]:
                assert '_' not in word


                word_inflections = getAllInflections(word)
                if not word_inflections:
                    word_inflections = dict(getAllInflectionsOOV(word,'VERB'), **getAllInflectionsOOV(word,'NOUN'))

                word_l_inflections = []
                for word_inflection in list(word_inflections.values()):
                    word_l_inflections.extend(list(word_inflection))

                word_l_inflections = list(set(word_l_inflections))
                word_l = [str(word)]

                # 一个premise 多少个tail
                # 一个tail 要sample * conjunction_word 次数

                for conjuntion_word in conjunction_words:
                    conjunction_l = []
                    conjunction_l.append(str(conjuntion_word))

                    all_l = []
                    all_l.append(conjunction_l)
                    all_l.append(word_l)

                    fc.write(str(all_l).replace("'","\""))
                    fc.write('\n')



                    all_l_inflect = []
                    all_l_inflect.append(conjunction_l)
                    all_l_inflect.append(word_l_inflections)


                    fc_inflect.write(str(all_l_inflect).replace("'","\""))
                    fc_inflect.write('\n')


                    if negation_words is not None:
                        all_l = []
                        all_l.append(conjunction_l)
                        all_l.append(word_l)
                        all_l.append(["not","no"])

                        fc.write(str(all_l).replace("'","\""))
                        fc.write('\n')

                        all_l_inflect = []
                        all_l_inflect.append(conjunction_l)
                        all_l_inflect.append(word_l_inflections)
                        all_l_inflect.append(["not","no"])

                        fc_inflect.write(str(all_l_inflect).replace("'","\""))
                        fc_inflect.write('\n')


    fi.close()
    fc.close()
    fc_inflect.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate related word')
    parser.add_argument('--input_file', default= 'input_file/query_CPE.csv')
    parser.add_argument('--save_file', default= './related_words.pkl')
    parser.add_argument('--model_name')
    parser.add_argument('--force_generate', help='whether force to regenerate related words' ,action='store_true')
    parser.add_argument('--csv_file', help='the file where we find the conclusion of the input',default='./data/ATOMIC10X_filter.csv')
    arts = parser.parse_args()
    main(arts)