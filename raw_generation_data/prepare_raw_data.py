import copy
import pickle
from helper import *

from get_related_words import get_related_words
import random
import os
from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV
import argparse
from split_train_infer import split_train_infer


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


def back_to_sen_mask(head,relation,tail,model_name,transition_word):
    if 'bart' in model_name:
        mask = '<mask>'
    elif 't5' in model_name:
        mask = '<extra_id_0>'


    if relation == 'xAttr':
        if transition_word == 'and':
            content = head + ' and ' + mask + ', so PersonX is seen as ' + tail + '.'
        elif transition_word == 'while':
            content = head + ' while ' + mask + ', so PersonX is seen as ' + tail + '.'
        elif transition_word == 'but':
            content = mask + ' but ' + head + ', so PersonX is seen as ' + tail + '.'
        elif transition_word == 'although':
            content = 'Although ' +  mask + ', ' + head + ', so PersonX is seen as ' + tail + '.'

    elif relation == 'xReact':
        if transition_word == 'and':
            content = head + ' and ' + mask + ', so PersonX feels ' + tail + '.'
        elif transition_word == 'while':
            content = head + ' while ' + mask + ', so PersonX feels ' + tail + '.'
        elif transition_word == 'but':
            content = mask + ' but ' + head + ', so PersonX feels ' + tail + '.'
        elif transition_word == 'although':
            content = 'Although ' +  mask + ', ' + head + ', so PersonX feels ' + tail + '.'

    return content


def main(args):

    transition_words = ["and","while","but","although"]
    needed_relations = {'xReact','xAttr'}
    num_of_cons = 5

    model_name = args.model_name

    premise_inputs = dict()
    premise_orders = dict()
    premise_constraints = dict()



    if not os.path.exists(args.save_file) or args.force == True:

        get_related_words(args.input_file,args.save_file)


    with open(args.save_file,'rb') as f:
        premise_words = pickle.load(f)


    all_tuples_file = args.csv_file

    df = pd.read_csv(all_tuples_file,names = ['head','relation','tail','order'],index_col='head')


    for query_index,query in enumerate(premise_words.keys()):
        df_selected = df.loc[query]
        relations = list(df_selected['relation'])
        tails = list(df_selected['tail'])
        orders = list(df_selected['order'])

        desired_relations = list(set(relations) & needed_relations)


        if len(desired_relations) == 0:
            continue

        relations_tails = []
        rt_orders = []
        count = 0
        for index,relation in enumerate(relations):
            if relation in desired_relations:
                relations_tails.append((relation,tails[index]))
                rt_orders.append(orders[index])

                count += 1;
                desired_relations.remove(relation)

            if len(desired_relations) == 0:
                break

            if count == 2:
                break


        backed_sents = []
        bs_orders = []
        for index,relation_tail in enumerate(relations_tails):
            relation,tail = relation_tail[0],relation_tail[1]

            for transition_word in transition_words:
                backed_sent = back_to_sen_mask(query,relation,tail,model_name,transition_word)
                backed_sents.append(backed_sent)

                bs_orders.append(f'{rt_orders[index]}_{transition_word}')


        premise_inputs[query] = backed_sents
        premise_orders[query] = bs_orders




        words_candidates = premise_words[query]

        words_candidates = random.sample(words_candidates,num_of_cons)

        assert len(words_candidates) == num_of_cons

        premise_constraints[query] = words_candidates




    negation_words = ['no']


    fi_write = []
    fc_lemma_write = []
    fc_inflect_write = []

    # dis short for disjunction

    fi_write_dis = []
    fc_lemma_write_dis = []
    fc_inflect_write_dis = []


    for premise in premise_constraints.keys():


        premise_constraints_length = len(premise_constraints[premise])

        # premise_inputs
        for backed_sent,bs_order in zip(premise_inputs[premise],premise_orders[premise]):

            for _ in range(premise_constraints_length):
                fi_write.append((backed_sent,bs_order))
                fi_write.append((backed_sent,f'{bs_order}_neg'))


            fi_write_dis.append((backed_sent,bs_order))
            fi_write_dis.append((backed_sent,f'{bs_order}_neg'))


        for _ in range(len(premise_inputs[premise])):

            word_inflections_dis = []
            word_lemmas_dis = []

            for word in premise_constraints[premise]:

                word_inflections = getAllInflections(word)
                if not word_inflections or len(word_inflections) == 0:
                    word_inflections = dict(getAllInflectionsOOV(word,'VERB'), **getAllInflectionsOOV(word,'NOUN'))

                word_inflections_l = []
                for word_inflection in list(word_inflections.values()):
                    word_inflections_l.extend(list(word_inflection))

                word_inflections = list(set(word_inflections_l))
                word_lemmas = [word]
                word_inflections_dis.append(word_inflections)
                word_lemmas_dis.append(word_lemmas)

                fc_lemma_write.append([word_lemmas])
                fc_inflect_write.append([word_inflections])

                if negation_words is not None:
                    negation_inflections = ["not","no"]
                    negation_lemmas = ["no"]

                    fc_lemma_write.append([word_lemmas,negation_lemmas])
                    fc_inflect_write.append([word_inflections,negation_inflections])

            tmp = []
            for _ in word_lemmas_dis:
                tmp += _
            word_lemmas_dis = [tmp]

            word_lemmas_dis_copy = copy.deepcopy(word_lemmas_dis)
            word_lemmas_dis.append(negation_lemmas)

            tmp = []
            for _ in word_inflections_dis:
                tmp += _
            word_inflections_dis = [tmp]

            word_inflections_dis_copy = copy.deepcopy(word_inflections_dis)
            word_inflections_dis.append(negation_inflections)


            fc_lemma_write_dis.append(word_lemmas_dis_copy)
            fc_lemma_write_dis.append(word_lemmas_dis)

            fc_inflect_write_dis.append(word_inflections_dis_copy)
            fc_inflect_write_dis.append(word_inflections_dis)



    len(fc_inflect_write_dis) == len(fc_lemma_write_dis) == len(fi_write_dis)
    len(fc_inflect_write) == len(fc_lemma_write) == len(fi_write)


    num_variations = (premise_constraints_length * (len(negation_words) + 1) * len(transition_words) * len(needed_relations))
    num_variations_dis = ((len(negation_words) + 1) * len(transition_words) * len(needed_relations))

    total_groups_train = 6
    total_groups_train_dis = 12

    groups_for_train = random.sample(range(int(len(fi_write)/num_variations)) ,total_groups_train)
    groups_for_train_dis = random.sample(range(int(len(fi_write_dis)/num_variations_dis)) ,total_groups_train_dis)


    split_args = dict()
    split_args_dis = dict()

    split_args['fi_write'] = fi_write
    split_args['fc_lemma_write'] = fc_lemma_write
    split_args['fc_inflect_write'] = fc_inflect_write
    split_args['num_variations'] = num_variations
    split_args['groups_for_train'] = groups_for_train
    split_args['output_file'] = args.output_file
    split_args['model_name'] = model_name

    split_args_dis['fi_write'] = fi_write_dis
    split_args_dis['fc_lemma_write'] = fc_lemma_write_dis
    split_args_dis['fc_inflect_write'] = fc_inflect_write_dis
    split_args_dis['num_variations'] = num_variations_dis
    split_args_dis['groups_for_train'] = groups_for_train_dis
    split_args_dis['output_file'] = args.output_file
    split_args_dis['model_name'] = model_name

    split_train_infer(split_args)
    split_train_infer(split_args_dis,for_dis=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate related word')
    parser.add_argument('--input_file', default= '../longtail_data/input.csv')
    parser.add_argument('--save_file', default= './related_words.pkl')
    parser.add_argument('--model_name', choices = ['bart','t5'], default='t5')
    parser.add_argument('--force', help='whether force to regenerate related words' ,action='store_true')
    parser.add_argument('--csv_file', help='the file where we find the conclusion of the input',default='../longtail_data/ATOMIC10X_filter.csv')
    parser.add_argument('--output_file', default= '../longtail_data')
    args = parser.parse_args()
    main(args)