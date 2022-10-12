
from GPT3_generation import *
import time
import backoff
import argparse
from tqdm import tqdm
import jsonlines



def main(args):

    demonstration_conti = \
    "Input: PersonX feels creative, because PersonX writes poems in her diary and [mask].\n"\
    "Constraint: [pillow, shoes, cloth, box, emotion]\n"\
    "Output: PersonX feels creative, because PersonX writes poems in her diary and expresses her emotion.\n"\
    "Input: PersonX is seen as fashionable because PersonX acquires an expensive shirt despite the fact that [mask].\n"\
    "Constraint: [mouse, computer, window, pen, house]\n"\
    "Output: PersonX is seen as fashionable, because PersonX acquires an expensive shirt despite the fact that PersonX lives in a poor house.\n"\
    "Input: PersonX feels uncertain, because PersonX asks what to do while [mask].\n"\
    "Constraint: [busy, school, bus, rice, meat]\n"\
    "Output:PersonX feels uncertain, because PersonX asks what to do while being busy with trivial stuff.\n"\
    "Input: PersonX feels confident because PersonX tells the whole truth even though the fact that [mask].\n"\
    "Constraint: [bottle, water, medicine, hard, tissue]\n"\
    "Output: PersonX feels confident because PersonX tells the whole truth even though the fact that problem is hard.\n"\


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

        reader = csv.reader(f)
        inputs = []
        inputs_order = []
        for line in reader:
            inputs.append(line[0].replace(original_mask,'[mask]'))
            inputs_order.append(line[1])


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
    breakpoint()


    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def gpt_generate(input,inflection_constraint,lemma_constraint):
        return gpt3_wrapper.prompt_generation(input,inflection_constraint,lemma_constraint)

    gpt3_wrapper = PromptWrapper(demonstration,args.no_filter)
    print('We use gpt3 to do generation')
    for index,(input,inflection_constraint,lemma_constraint) in enumerate(tqdm(list(zip(inputs,inflection_constraints,lemma_constraints)))):

        generation = gpt_generate(input,inflection_constraint,lemma_constraint)
        generations.append(generation)

    assert len(inputs) == len(generations) == len(inputs_order)

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
    parser.add_argument('--inputs')
    parser.add_argument('--lemma_constraints')
    parser.add_argument('--inflection_constraints')
    parser.add_argument('--outputs')
    parser.add_argument('--model_type', choices = ['t5','bart'], default = 't5')
    parser.add_argument('--no_filter', action='store_true')
    parser.add_argument('--Mturk', action='store_true')
    parser.add_argument('--num_groups', default= -1, type=int)
    parser.add_argument('--variations_per_group', default= -1, type=int)
    args = parser.parse_args()
    main(args)