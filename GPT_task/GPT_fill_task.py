
import os
from pathlib import Path
from GPT3_generation import *
import time
import backoff
import argparse
from tqdm import tqdm
import pickle
import jsonlines
import json
import csv
from get_data_utils import All_Data

# random.seed(42)






demonstration_conti = \
"Input: PersonX feels creative because PersonX writes poems in her diary and [mask].\n"\
"Constraint: [pillow, shoes, cloth, box, emotion]\n"\
"Output: PersonX feels creative because PersonX writes poems in her diary and expresses her emotion.\n"\
"Input: PersonX is seen as intelligent because PersonX is a student in PersonY's course and [mask].\n"\
"Constraint: [normal, grade, place, plane, bed], [no]\n"\
"Output: PersonX is seen as intelligent because PersonX is a student in PersonY's course and no one gets higher score than PersonX.\n"\
"Input: PersonX is seen as fashionable because PersonX acquires an expensive shirt despite the fact that [mask].\n"\
"Constraint: [mouse, computer, window, pen, house]\n"\
"Output: PersonX is seen as fashionable because PersonX acquires an expensive shirt despite the fact that PersonX lives in a poor house.\n"\
"Input: PersonX is seen as diligent because PersonX completes her paper despite the fact that [mask].\n"\
"Constraint: [mentor, grocery, hand, drug, skin], [no]\n"\
"Output: PersonX is seen as diligent because PersonX completes her paper despite the fact that no mentor guides PersonX.\n"\
"Input: PersonX feels uncertain because PersonX asks what to do while [mask].\n"\
"Constraint: [busy, school, bus, rice, meat]\n"\
"Output:PersonX feels uncertain because PersonX asks what to do while being busy with trivial stuff.\n"\
"Input: PersonX feels nervous because PersonX sneaks into PersonY's room while [mask].\n"\
"Constraint: [purse, mountain, water, disk, bowl], [no]\n"\
"Output: PersonX feels nervous because PersonX sneaks into PersonY's room while does not find purse.\n"\
"Input: PersonX feels pleased because PersonX gets a job at club even though [mask].\n"\
"Constraint: [road, team, car, tree, street]\n"\
"Output: PersonX feels pleased because PersonX gets a job at club even though PersonX dismissed the team.\n"\
"Input: PersonX feels confident because PersonX tells the whole truth even though [mask].\n"\
"Constraint: [bottle, water, medicine, easy, tissue], [no]\n"\
"Output: PersonX feels confident because PersonX tells the whole truth even though problem is not easy."




demonstration = \
"Input: PersonX writes poems in her diary and [mask], so PersonX feels creative.\n"\
"Constraint: [pillow, shoes, cloth, box, emotion]\n"\
"Output: PersonX writes poems in her diary and expresses her emotion, so PersonX feels creative.\n"\
"Input: PersonX is a student in PersonY's course and [mask], so PersonX is seen as intelligent.\n"\
"Constraint: [normal, grade, place, plane, bed], [no]\n"\
"Output: PersonX is a student in PersonY's course and no one gets higher grade than him, so PersonX is seen as intelligent.\n"\
"Input: [mask] but PersonX acquires an expensive shirt, so PersonX is seen as fashionable.\n"\
"Constraint: [mouse, computer, window, pen, house]\n"\
"Output: PersonX lives in a poor house but PersonX acquires an expensive shirt, so PersonX is seen as fashionable.\n"\
"Input: [mask] but PersonX completes her paper, so PersonX is seen as diligent.\n"\
"Constraint: [mentor, grocery, hand, drug, skin], [no]\n"\
"Output: No mentor guides PersonX but PersonX completes her paper, so PersonX is seen as diligent.\n"\
"Input: PersonX asks what to do while [mask], so PersonX feels uncertain.\n"\
"Constraint: [busy, school, bus, rice, meat]\n"\
"Output: PersonX asks what to do while being busy with trivial stuffs, so PersonX feels uncertain.\n"\
"Input: PersonX sneaks into PersonY's room while [mask], so PersonX feels nervous.\n"\
"Constraint: [purse, mountain, water, disk, bowl], [no]\n"\
"Output: PersonX sneaks into PersonY's room while does not find the purse, so PersonX feels nervous.\n"\
"Input: Although [mask], PersonX gets a job at club, so PersonX feels pleased.\n"\
"Constraint: [road, team, car, tree, street]\n"\
"Output: Although PersonX dismissed the team, PersonX gets a job at club, so PersonX feels pleased.\n"\
"Input: Although [mask], PersonX tells the whole truth, so PersonX feels confident.\n"\
"Constraint: [bottle, water, medicine, easy, tissue], [no]\n"\
"Output: Although problem is not easy, PersonX tells the whole truth, so PersonX feels confident."


# This is for property-centric
# increase, decrease
# I I D D D D I I
# negation , w/o negation
#  T F T F T T T F
# length of cons
# use of cons words

demonstration_conti = \
"Input: The power of person will increase because personX eats food and [mask].\n"\
"Constraint: [exercise, run, hit]\n"\
"Output: The power of person will increase because personX eats food and do exercises by running.\n"\
"Input: The price of diamond will increase because personX takes care of the diamond and [mask].\n"\
"Constraint: [couple, marriage, market, money], [no]\n"\
"Output: The price of diamond will increase because personX takes care of the diamond and no more diamonds are available at market.\n"\
"Input: The weight of dog will decrease because personX feeds the dog while [mask].\n"\
"Constraint: [food, water, toy, cage, sick, ground]\n"\
"Output: The weight of dog will decrease because personX feeds the dog while the dog is sick.\n"\
"Input: The temperature of water will decrease because personX boils water while [mask].\n"\
"Constraint: [air, resource, medicine, heat, bag], [no]\n"\
"Output: The temperature of water will decrease because personX boils water while does not have enough heat resources.\n"\
"Input: The length of ruler will decrease because personX cuts off the ruler even though [mask].\n"\
"Constraint: [store, new, math]\n"\
"Output: The length of ruler will decrease because personX cuts off the ruler even though it is bought from store newly.\n"\
"Input: The resistance of wind will decrease because personX closes the window even though [mask].\n"\
"Constraint: [river, tree, grass, stone]\n"\
"Output: The resistance of wind will decrease because personX closes the window even though stones at grass are blowed up.\n"\

"Input: The speed of bike will increase because personX drives down hill despite the fact that [mask].\n"\
"Constraint: [down ,car, tree, brake]\n"\
"Output: The speed of bike will increase because personX drives down hill despite the fact that she tries to brake.\n"\

"Input: The height of person will increase because personX grows up despite the fact that [mask].\n"\
"Constraint: [sport, medicine, ball, rice], [no]\n"\
"Output: The height of person will increase because personX grows up despite the fact that he does not play basketball."\



demonstration_conti = \
"Input: The depth of ocean will increase because personX increase the ocean's salinity and [mask].\n"\
"Constraint: [pump], [well]\n"\
"Output: The depth of ocean will increase because personX increase the ocean's salinity and pump from well. \n"\
"Input: The energy of wind will increase becasue the wind blow harder and [mask].\n"\
"Constraint: [breathe], [sports, people], [no]\n"\
"Output: The energy of wind will increase becasue the wind blow harder and no people breathe hard.\n"\
"Input: The force of ice will decrease bacause the ice be hit while [mask].\n"\
"Constraint: [kick], [soccer, ball, playground]\n"\
"Output: The force of ice will decrease bacause the ice be hit while children kick the soccer ball.\n"\
"Input: The frequency of of bowel movements will decrease bacause PersonX eat less while [mask].\n"\
"Constraint: [eat], [food, salt], [no]\n"\
"Output: The frequency of of bowel movements will decrease bacause PersonX eat less while does not eat salty food.\n"\
"Input: The depth of swimming pool will decrease because personX drain the water even thouhgh [mask].\n"\
"Constraint: [increase], [valley, wall, air]\n"\
"Output: The depth of swimming pool will decrease because personX drain the water even thouhgh increasing height of wall.\n"\
"Input: The electric current of blender will decrease because personX unplug the blender even though [mask].\n"\
"Constraint: [charge], [machine, clothes]\n"\
"Output: The electric current of blender will decrease because personX unplug the blender even though personX charges the washing machine.\n"\
"Input: The height of balloon will increase because the balloon be fill with air despite the fact that [mask].\n"\
"Constraint: [undermine], [bank, river]\n"\
"Output: The height of balloon will increase because the balloon be fill with air despite the fact that bank river is undermined.\n"\
"Input: The pressure of heart rate will increase because personX be in a hot environment despite the fact that [mask].\n"\
"Constraint: [run], [track, hill], [no]\n"\
"Output: The pressure of heart rate will increase because personX be in a hot environment despite the fact that he does not run on the track."



# ****************# ****************# ****************# ****************# ****************# ****************# ****************# ****************
# ****************# ****************# ****************# ****************# ****************# ****************# ****************# ****************

# ****************# ****************# ****************# ****************# ****************# ****************# ****************# ****************
demonstration_conti = \
"Input: The electric current of blender will decrease because personX unplug the blender and [mask].\n"\
"Constraint: [charge], [machine, clothes, electricity], [machine, clothes, floor], [machine, electricity, floor], [clothes, electricity, floor]\n"\
"Output: The electric current of blender will decrease because personX unplug the blender and personX charges the washing machine.\n"\
"Input: The energy of wind will increase becasue the wind blow harder and [mask].\n"\
"Constraint: [breathe], [sports, tennis, people], [sports, tennis, kids], [sports, people, kids], [tennis, people, kids], [no]\n"\
"Output: The energy of wind will increase becasue the wind blow harder and no people breathe hard.\n"\
"Input: The depth of ocean will increase because personX increase the ocean's salinity and [mask].\n"\
"Constraint: [pump], [well], [machine]\n"\
"Output: The depth of ocean will increase because personX increase the ocean's salinity and pump the well by machine. \n"\
"Input: The frequency of of bowel movements will decrease bacause PersonX eat less and [mask].\n"\
"Constraint: [eat], [food], [salt], [no]\n"\
"Output: The frequency of of bowel movements will decrease bacause PersonX eat less and does not eat salty food.\n"\
"Input: The pressure of heart rate will increase because personX be in a hot environment and [mask].\n"\
"Constraint: [run], [track, hill], [track, grass], [hill, grass], [no]\n"\
"Output: The pressure of heart rate will increase because personX be in a hot environment and he does not run on the track of the hill\n"\
"Input: The force of ice will decrease bacause the ice be hit and [mask].\n"\
"Constraint: [kick, run], [soccer, ball], [playground, ball], [soccer, playground]\n"\
"Output: The force of ice will decrease bacause the ice be hit and children kick the soccer on the playground."



demonstration_conti = \
"Constraint: [charge], [machine, clothes, electricity], [machine, clothes, floor], [machine, electricity, floor], [clothes, electricity, floor] ; Input: The electric current of blender will decrease because you unplug the blender and ; Output:\n"\
"Output: The electric current of blender will decrease because you unplug the blender and you charge the washing machine.\n"\
"Constraint: [breathe], [sports, tennis, people], [sports, tennis, kids], [sports, people, kids], [tennis, people, kids], [no] ; Input: The energy of wind will increase becasue the you blow harder and ; Output:\n"\
"Output: The energy of wind will increase becasue the you blow harder and no people breathe hard.\n"\
"Constraint: [pump], [well], [machine] ; Input: The depth of ocean will increase because you increase the ocean's salinity and ; Output:\n"\
"Output: The depth of ocean will increase because you increase the ocean's salinity and you pump the well by machine.\n"\
"Constraint: [eat], [food], [salt], [no] ; Input: The frequency of of bowel movements will decrease bacause you eat less and ; Output:\n"\
"Output: The frequency of of bowel movements will decrease bacause you eat less and you do not eat salty food.\n"\
"Constraint: [run], [track, hill], [track, grass], [hill, grass], [no] ; Input: The pressure of heart rate will increase because you is in a hot environment and ; Output:\n"\
"Output: The pressure of heart rate will increase because you is in a hot environment and you do not run on the track of the hill.\n"\
"Constraint: [kick, run], [soccer, ball], [playground, ball], [soccer, playground] ; Input: The force of ice will decrease bacause you hit the ice and ; Output:\n"\
"Output: The force of ice will decrease bacause you hit the ice and you kick the soccer on the playground."


def change_format(x):
    cons_string = ''
    for tmp in x:
        tmp = '[' + ', '.join(tmp) + ']'
        cons_string += str(tmp) + ', '
    cons_string = cons_string[:-2]
    
    return cons_string


def main(args):
    all_data = All_Data(args.all_data)

    print(f'We load data ids from {args.inputs}')
    with open(args.inputs) as f:
        ids = []
        reader = csv.reader(f)
        for line in reader:
            ids.extend(line)
        
    datas = all_data.get_data(ids)


    mask = '[mask]'

    inputs = []
    
    for line in datas['conti_templates']:
        inputs.append(line.replace(f'{mask}.','').strip())
    lemmas = list(map(change_format,datas['lemmas']))
    inflections = datas['inflections']

    if args.n_obs != -1:
        inputs = inputs[:args.n_obs]
        inflections = inflections[:args.n_obs]
        lemmas = lemmas[:args.n_obs]

    assert len(inputs) == len(lemmas) == len(inflections)


    generations = []
    ids_new = []


    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def gpt_generate(input,inflection,lemma):
        return gpt3_wrapper.prompt_generation(input,inflection,lemma)

    if args.conti:
        demonstration = demonstration_conti
    else:
        raise NotImplementedError


    gpt3_wrapper = PromptWrapper(demonstration,args.no_filter)
    
    for index,(input,inflection,lemma) in enumerate(tqdm(list(zip(inputs,inflections,lemmas)))):

        generations_gpt = gpt_generate(input,inflection,lemma)

        if args.no_filter:
            if len(generations_gpt) == 0:
                generations_gpt = ['Head_dont_match']

        if len(generations_gpt) != 0:
            final_len = len(generations_gpt[:args.needed_count])
            generations.extend(generations_gpt[:args.needed_count])
            ids_new.extend([ids[index]] * final_len)
        else:
            print(f'For {input}, we dont have qualified generation')


    assert len(generations) == len(ids_new)

    Path(args.outputs).mkdir(parents= True,exist_ok=True)

    if not args.no_filter:
        outputs = Path(args.outputs) / 'gpt_outputs.jsonl'

        with jsonlines.open(outputs,'w') as f:
            for index in range(len(generations)):
                tmp = dict()
                
                tmp['generation'] = generations[index]
                tmp['id'] = ids_new[index]
                f.write(tmp)
    else:
        outputs = Path(args.outputs) / 'gpt3.csv'
        with open(outputs,'w') as f:
            writer = csv.writer(f)
            writer.writerows(list(zip(generations,ids_new)))
    print(f'We save data at {outputs}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gpt3 generation')
    parser.add_argument('--all_data', default = '/home/zeyi/longtail/property_centric_process/all_data.pkl')
    parser.add_argument('--inputs')
    parser.add_argument('--outputs')
    parser.add_argument('--needed_count', type = int, help = 'How many generations you want for each example')
    parser.add_argument('--conti', action='store_true')

    parser.add_argument('--no_filter', action='store_true')
    parser.add_argument('--n_obs', default= -1, type=int)
    args = parser.parse_args()
    main(args)