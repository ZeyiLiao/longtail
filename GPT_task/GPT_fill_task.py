
from email.policy import default
from GPT3_generation import *
import time
import backoff
import argparse
from argparse import ArgumentParser


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
    "Constraint:help, no\n"\
    "Output: Although no one help, PersonX asks what to do, so PersonX feels uncertain."


    print(demonstration)

    gpt3_wrapper = PromptWrapper(demonstration,args.no_filter)


    def change_format(x):
        return x.replace('[','').replace(']','').replace("\"","").replace(', not','')


    with open (args.input) as f:
        inputs = [x.rstrip() for x in f.readlines()]


    with open (args.constraint_lemma) as f:
        constraints = [x.rstrip() for x in f.readlines()]
        constraints = list(map(change_format,constraints))


    generations = []

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def gpt_generate(input,constraint):
        return gpt3_wrapper.prompt_generation(input,constraint)

    for index,(input,constraint) in enumerate(list(zip(inputs,constraints))):
        # if index != 0 and index % 5 == 0:
        #     print('sleep for 60 secs')
        #     time.sleep(60)

        generation = gpt_generate(input,constraint)
        generations.append(generation)



    nl = '\n'
    if not args.Mturk:
        with open(args.output,'w') as f:
            f.write('GPT_fill results')
            f.write(nl)
            f.write('demonstration:')
            f.write(nl)
            f.write(demonstration)
            f.write(nl)
            f.write(nl)
            f.write(nl)
            for index in range(len(generations)):
                input = inputs[index]
                constraint = constraints[index]
                generation = generations[index]
                f.write(f'input: {input}')
                f.write(nl)
                f.write(f'constraint: {constraint}')
                f.write(nl)
                f.write('output:')
                f.write(nl)
                f.write(f'{generation}')
                f.write(nl)
                f.write(nl)
                f.write('***************************')
                f.write(nl)

    else:
        with open(args.output,'w') as f:
            for index in range(len(generations)):
                generation = generations[index]
                f.write(f'{generation}')
                f.write(nl)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gpt3 generation')
    parser.add_argument('--input',default = 'GPT_fill_input.txt')
    parser.add_argument('--constraint_lemma',default = 'GPT_fill_constraints.txt')
    parser.add_argument('--output', default = 'GPT_fill_output.txt')
    parser.add_argument('--no_filter', action='store_true')
    parser.add_argument('--Mturk', action='store_true')
    args = parser.parse_args()
    main(args)