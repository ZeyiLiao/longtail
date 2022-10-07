
from GPT3_generation import *
import time
import backoff
import argparse
from tqdm import tqdm
import jsonlines


def main(args):

    demonstration = \
    "Input: PersonX sneaks into PersonY's room and [mask], so PersonX feels nervous.\n"\
    "Constraint: closet\n"\
    "Output: PersonX sneaks into PersonY's room and sees a closet space, so PersonX feels nervous.\n"\
    "Input: PersonX sneaks into PersonY's room and [mask], so PersonX feels nervous.\n"\
    "Constraint: furniture, no\n"\
    "Output: PersonX sneaks into PersonY's room and does not find furniture, so PersonX feels nervous.\n"\
    "Input: [mask] but PersonX sneaks into PersonY's room, so PersonX feels nervous.\n"\
    "Constraint: police\n"\
    "Output: Police surround the house but PersonX sneaks into PersonY's room, so PersonX feels nervous.\n"\
    "Input: [mask] but PersonX sneaks into PersonY's room, so PersonX feels nervous.\n"\
    "Constraint: money, no\n"\
    "Output: PersonX does not need money but PersonX sneaks into PersonY's room, so PersonX feels nervous\n"\
    "Input: PersonX asks what to do while [mask], so PersonX feels uncertain.\n"\
    "Constraint: seek\n"\
    "Output: PersonX asks what to do while seeks suggestions, so PersonX feels uncertain.\n"\
    "Input: PersonX asks what to do while [mask], so PersonX feels uncertain.\n"\
    "Constraint: help, no\n"\
    "Output: PersonX asks what to do while no one help, so PersonX feels uncertain.\n"\
    "Input: Although [mask], PersonX asks what to do, so PersonX feels uncertain.\n"\
    "Constraint: consultant\n"\
    "Output: Although consultant sleeps, PersonX asks what to do, so PersonX feels uncertain.\n"\
    "Input: Although [mask], PersonX asks what to do, so PersonX feels uncertain.\n"\
    "Constraint: help, no\n"\
    "Output: Although no one help, PersonX asks what to do, so PersonX feels uncertain."


    gpt3_wrapper = PromptWrapper(demonstration,args.no_filter)


    def change_format(x):
        tmp = []
        for _ in x:
            tmp.extend(_)
        try:
            tmp.remove('not')
        finally:
            string = ', '.join(tmp)
            return string


    with open (args.input) as f:
        if args.model_type == 't5':
            original_mask = '<extra_id_0>'
        else:
            raise NotImplementedError

        inputs = [x.rstrip().replace(original_mask,'[mask]') for x in f.readlines()]

    with open (args.constraint_lemma) as f:
        constraints = [json.loads(x) for x in f.readlines()]
        constraints = list(map(change_format,constraints))


    if args.num_groups != -1:
        inputs = inputs[:32 * args.num_groups]
        constraints = constraints[:32 * args.num_groups]
    assert len(inputs) == len(constraints)


    generations = []

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def gpt_generate(input,constraint):
        return gpt3_wrapper.prompt_generation(input,constraint)

    for index,(input,constraint) in enumerate(tqdm(list(zip(inputs,constraints)))):

        generation = gpt_generate(input,constraint)
        generations.append(generation)

    assert len(inputs) == len(generations)

    Path(args.output).mkdir(parents= True,exist_ok=True)


    nl = '\n'
    if not args.Mturk:
        output = Path(args.output) / 'gpt_output.jsonl'
        with jsonlines.open(output,'w') as f:
            for index in range(len(generations)):
                tmp = dict()
                tmp['input'] = str(inputs[index])
                tmp['constraint'] = str(constraints[index])
                tmp['generation'] = str(generations[index])
                f.write(tmp)




    else:
        output = Path(args.output) / 'gpt_output.txt'
        with open(output,'w') as f:
            for index in range(len(generations)):
                generation = generations[index]
                if 'Input' in generation and 'Constraint' in generation:
                    generation = generation.split('\n')[0]
                f.write(f'{generation}')
                f.write(nl)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gpt3 generation')
    parser.add_argument('--input',default = '../longtail_data/raw_data/generation_input_t5_train.txt')
    parser.add_argument('--constraint_lemma',default = '../longtail_data/raw_data/generation_constraints_lemmas_t5_train.json')
    parser.add_argument('--output', default = 'GPT_fill_output.txt')
    parser.add_argument('--model_type', choices = ['t5','bart'], default = 't5')
    parser.add_argument('--no_filter', action='store_true')
    parser.add_argument('--Mturk', action='store_true')
    parser.add_argument('--num_groups', default= -1, type=int)
    args = parser.parse_args()
    main(args)