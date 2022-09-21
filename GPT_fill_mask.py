
from negation_prompt import *
import time

demonstration = \
"Input: PersonX sneaks into PersonY's room [mask] so PersonX feels nervous.\n"\
"Constraint: and, closet\n"\
"Output: PersonX sneaks into PersonY's room and sees a closet space, so PersonX feels nervous.\n"\
"Input: PersonX sneaks into PersonY's room [mask] so PersonX feels nervous.\n"\
"Constraint: and, furniture, no\n"\
"Output: PersonX sneaks into PersonY's room and does not find furniture, so PersonX feels nervous.\n"\
"Input: PersonX asks what to do [mask] so PersonX feels uncertain.\n"\
"Constraint: and, seek\n"\
"Output: PersonX asks what to do and seeks suggestions, so PersonX feels uncertain.\n"\
"Input: PersonX asks what to do [mask] so PersonX feels uncertain.\n"\
"Constraint:and, help, no\n"\
"Output: PersonX asks what to do and no one help, so PersonX feels uncertain."

print(demonstration)

gpt3_wrapper = PromptWrapper(demonstration)

def change_format(x):
    return x.replace('[','').replace(']','').replace("\"","").replace(', not','')

with open ('GPT_fill_input.txt') as f:

    inputs = [x.rstrip() for x in f.readlines()]

with open ('GPT_fill_constraints.txt') as f:
    constraints = [x.rstrip() for x in f.readlines()]
    constraints = list(map(change_format,constraints))

generations = []


for index,(input,constraint) in enumerate(list(zip(inputs,constraints))):
    if index != 0 and index % 30 == 0:
        print('sleep for 60 secs')
        time.sleep(60)
    generation = gpt3_wrapper.prompt_generation(input,constraint)
    generations.append(generation)



nl = '\n'
with open('GPT_fill_output.txt','w') as f:
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
