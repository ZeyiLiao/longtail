from operator import is_
import pickle
from helper import *
from get_related_words import get_related_words
import random
import os
from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV
import nltk
import argparse


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



    model_name = args.model_name

    premise_inputs = dict()
    premise_constraints = dict()

    # if os.path.exists(generation_constraints_lemmas_path):
    #     os.remove(generation_constraints_lemmas_path)

    # if os.path.exists(generation_inputs_path):
    #     os.remove(generation_inputs_path)

    # if os.path.exists(generation_constraints_inflections_path):
    #     os.remove(generation_constraints_inflections_path)



    if not os.path.exists(args.save_file) or args.force == True:

        get_related_words(args.input_file,args.save_file)



    with open(args.save_file,'rb') as f:
        premise_words = pickle.load(f)





    all_tuples_file = args.csv_file

    df = pd.read_csv(all_tuples_file,names = ['head','relation','tail'],index_col='head')


    for query_index,query in enumerate(premise_words.keys()):
        df_selected = df.loc[query]
        relations = list(df_selected['relation'])
        tails = list(df_selected['tail'])
        desired_relations = list(set(relations) & needed_relations)


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
            for transition_word in transition_words:
                backed_sent = back_to_sen_mask(query,relation,tail,model_name,transition_word)
                backed_sents.append(backed_sent)

        premise_inputs[query] = backed_sents



        words_candidates = premise_words[query]

        sample_times = 2

        words_candidates = random.sample(words_candidates,sample_times)

        assert len(words_candidates) == 2

        premise_constraints[query] = words_candidates




    negation_words = ['no']




    fi_write = []
    fc_lemma_write = []
    fc_inflect_write = []
    for premise in premise_constraints.keys():


        # 一个句子有多少个变种
        premise_constraints_length = len(premise_constraints[premise])
        factor = premise_constraints_length * (len(negation_words) + 1)

        # premise_inputs
        for backed_sent in premise_inputs[premise]:
            for _ in range(factor):
                fi_write.append(backed_sent)

        # premise_inputs
        # TODO: need to change this shit code
        for _ in range(len(premise_inputs[premise])):

            for word in premise_constraints[premise]:



                word_inflections = getAllInflections(word)
                if not word_inflections or len(word_inflections) == 0:
                    word_inflections = dict(getAllInflectionsOOV(word,'VERB'), **getAllInflectionsOOV(word,'NOUN'))

                word_inflections_l = []
                for word_inflection in list(word_inflections.values()):
                    word_inflections_l.extend(list(word_inflection))

                word_inflections = list(set(word_inflections_l))
                word_lemmas = [word]

                # 一个premise 多少个tail
                # 一个tail 要sample * conjunction_word 次数



                all_lemmas = []
                all_lemmas.append(word_lemmas)

                fc_lemma_write.append(all_lemmas)


                all_inflects = []
                all_inflects.append(word_inflections)

                fc_inflect_write.append(all_inflects)


                if negation_words is not None:
                    all_lemmas = []

                    all_lemmas.append(word_lemmas)
                    all_lemmas.append(["not","no"])

                    fc_lemma_write.append(all_lemmas)

                    all_inflects = []

                    all_inflects.append(word_inflections)
                    all_inflects.append(["not","no"])

                    fc_inflect_write.append(all_inflects)




    len(fc_inflect_write) == len(fc_lemma_write) == len(fi_write)
    num_variations = (premise_constraints_length * (len(negation_words) + 1) * len(transition_words) * len(needed_relations))

    total_groups_train = 6
    groups_for_train = random.sample(range(int(len(fi_write)/num_variations)) ,total_groups_train)


    train_inputs = []
    infer_inputs = []
    train_lemmas = []
    infer_lemmas = []
    train_inflections = []
    infer_inflections = []


    for index in range(0,len(fc_inflect_write),num_variations):
        if index//num_variations in groups_for_train:
            train_inputs.extend(fi_write[index:index+num_variations])
            train_lemmas.extend(fc_lemma_write[index:index+num_variations])
            train_inflections.extend(fc_inflect_write[index:index+num_variations])
        else:
            infer_inputs.extend(fi_write[index:index+num_variations])
            infer_lemmas.extend(fc_lemma_write[index:index+num_variations])
            infer_inflections.extend(fc_inflect_write[index:index+num_variations])


    root_dir = f'{args.output_file}/raw_data'
    Path(root_dir).mkdir(parents= True,exist_ok=True)
    root_dir = Path(root_dir)



    train_or_infer = 'train'
    nl = '\n'

    generation_inputs_path = root_dir / f'generation_input_{model_name}_{train_or_infer}.txt'
    generation_constraints_lemmas_path = root_dir / f'generation_constraints_lemmas_{model_name}_{train_or_infer}.json'
    generation_constraints_inflections_path = root_dir / f'generation_constraints_inflections_{model_name}_{train_or_infer}.json'

    fi = open(generation_inputs_path,'w')
    fc_lemma = open(generation_constraints_lemmas_path,'w')
    fc_inflect = open(generation_constraints_inflections_path,'w')

    for index in range(len(train_inputs)):
        fi.write(train_inputs[index])
        fi.write(nl)
        json.dump(train_lemmas[index],fc_lemma)
        fc_lemma.write(nl)
        json.dump(train_inflections[index],fc_inflect)
        fc_inflect.write(nl)

    fi.close()
    fc_lemma.close()
    fc_inflect.close()

# **************************************************************************
    train_or_infer = 'infer'
    nl = '\n'

    generation_inputs_path = root_dir / f'generation_input_{model_name}_{train_or_infer}.txt'
    generation_constraints_lemmas_path = root_dir / f'generation_constraints_lemmas_{model_name}_{train_or_infer}.json'
    generation_constraints_inflections_path = root_dir / f'generation_constraints_inflections_{model_name}_{train_or_infer}.json'

    fi = open(generation_inputs_path,'w')
    fc_lemma = open(generation_constraints_lemmas_path,'w')
    fc_inflect = open(generation_constraints_inflections_path,'w')

    for index in range(len(infer_inputs)):
        fi.write(infer_inputs[index])
        fi.write(nl)
        json.dump(infer_lemmas[index],fc_lemma)
        fc_lemma.write(nl)
        json.dump(infer_inflections[index],fc_inflect)
        fc_inflect.write(nl)

    fi.close()
    fc_lemma.close()
    fc_inflect.close()



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