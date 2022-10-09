
from GPT3_generation import *
import time
import backoff
import argparse
from tqdm import tqdm
import jsonlines


def main(args):

    demonstration = \
    "Input: PersonX sneaks into PersonY's room and [mask], so PersonX feels nervous.\n"\
    "Constraint: [closet, normal, place, plane, bed]\n"\
    "Output: PersonX sneaks into PersonY's room and sees a closet space, so PersonX feels nervous.\n"\
    "Input: PersonX sneaks into PersonY's room and [mask], so PersonX feels nervous.\n"\
    "Constraint: [furniture, mountain, water, disk, bowl], [no]\n"\
    "Output: PersonX sneaks into PersonY's room and does not find furnitures, so PersonX feels nervous.\n"\
    "Input: [mask] but PersonX sneaks into PersonY's room, so PersonX feels nervous.\n"\
    "Constraint: [police, computer, window, pen, mouse]\n"\
    "Output: Police surround the house but PersonX sneaks into PersonY's room, so PersonX feels nervous.\n"\
    "Input: [mask] but PersonX sneaks into PersonY's room, so PersonX feels nervous.\n"\
    "Constraint: [money, water, medicine, cigarette, tissue], [no]\n"\
    "Output: PersonX does not need money but PersonX sneaks into PersonY's room, so PersonX feels nervous\n"\
    "Input: PersonX asks what to do while [mask], so PersonX feels uncertain.\n"\
    "Constraint: [suggestion, shoes, cloth, box, pillow]\n"\
    "Output: PersonX asks what to do while seeks suggestions, so PersonX feels uncertain.\n"\
    "Input: PersonX asks what to do while [mask], so PersonX feels uncertain.\n"\
    "Constraint: [school, help, bus, rice, meat], [no]\n"\
    "Output: PersonX asks what to do while no one help, so PersonX feels uncertain.\n"\
    "Input: Although [mask], PersonX asks what to do, so PersonX feels uncertain.\n"\
    "Constraint: [road, street, car, tree, consultant]\n"\
    "Output: Although consultant sleeps, PersonX asks what to do, so PersonX feels uncertain.\n"\
    "Input: Although [mask], PersonX asks what to do, so PersonX feels uncertain.\n"\
    "Constraint: [help, grocery, hand, drug, skin], [no]\n"\
    "Output: Although no one help, PersonX asks what to do, so PersonX feels uncertain."


    def change_format(x):
        neg = []
        if len(x) == 2:
            neg = x[1]
        cons = x[0]
        tmp = []
        for _ in cons:
            tmp.append(_)
        random.shuffle(tmp)

        cons_string = '[' + ', '.join(tmp) + ']'
        if len(neg) != 0:
            cons_string = cons_string + f', [{neg[0]}]'

        return cons_string


    print(f'we load data from {args.inputs}')
    print(f'we load lemma constraints from {args.lemma_constraints}')
    print(f'we load inflection constraints from {args.inflection_constraints}')

    with open (args.inputs) as f:
        if args.model_type == 't5':
            original_mask = '<extra_id_0>'
        else:
            raise NotImplementedError

        inputs = [x.rstrip().replace(original_mask,'[mask]') for x in f.readlines()]


    with open (args.lemma_constraints) as f:
        lemma_constraints = [json.loads(x) for x in f.readlines()]
        lemma_constraints = list(map(change_format,lemma_constraints))

    with open (args.inflection_constraints) as f:
        inflection_constraints = [json.loads(x) for x in f.readlines()]



    if args.num_groups != -1:
        if args.variations_per_group == -1:
            raise NotImplementedError

        inputs = inputs[:args.variations_per_group * args.num_groups]
        inflection_constraints = inflection_constraints[:args.variations_per_group * args.num_groups]
        lemma_constraints = lemma_constraints[:args.variations_per_group * args.num_groups]


    assert len(inputs) == len(inflection_constraints) == len(lemma_constraints)


    generations = []


    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def gpt_generate(input,inflection_constraint,lemma_constraint):
        return gpt3_wrapper.prompt_generation(input,inflection_constraint,lemma_constraint)

    gpt3_wrapper = PromptWrapper(demonstration,args.no_filter)
    for index,(input,inflection_constraint,lemma_constraint) in enumerate(tqdm(list(zip(inputs,inflection_constraints,lemma_constraints)))):

        generation = gpt_generate(input,inflection_constraint,lemma_constraint)
        generations.append(generation)

    assert len(inputs) == len(generations)

    Path(args.outputs).mkdir(parents= True,exist_ok=True)



    nl = '\n'
    if not args.Mturk:
        outputs = Path(args.outputs) / 'gpt_outputs.jsonl'
        with jsonlines.open(outputs,'w') as f:
            for index in range(len(generations)):
                tmp = dict()
                tmp['input'] = str(inputs[index])
                tmp['constraint'] = str(lemma_constraints[index])
                tmp['generation'] = str(generations[index])
                f.write(tmp)


    else:
        outputs = Path(args.outputs) / 'gpt_outputs.txt'
        with open(outputs,'w') as f:
            for index in range(len(generations)):
                generation = generations[index]
                if 'Input' in generation and 'Constraint' in generation:
                    generation = generation.split('\n')[0]
                f.write(f'{generation}')
                f.write(nl)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gpt3 generation')
    parser.add_argument('--inputs',default = '../longtail_data/raw_data/input_t5_train.txt')
    parser.add_argument('--lemma_constraints',default = '../longtail_data/raw_data/lemma_constraints_t5_train.json')
    parser.add_argument('--inflection_constraints',default = '../longtail_data/raw_data/inflection_constraints_t5_train.json')
    parser.add_argument('--outputs', default = 'GPT_fill_output.txt')
    parser.add_argument('--model_type', choices = ['t5','bart'], default = 't5')
    parser.add_argument('--no_filter', action='store_true')
    parser.add_argument('--Mturk', action='store_true')
    parser.add_argument('--num_groups', default= -1, type=int)
    parser.add_argument('--variations_per_group', default= -1, type=int)
    args = parser.parse_args()
    main(args)