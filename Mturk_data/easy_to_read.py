inputs = []
gpt_outputs = []
neuro_outputs = []
vanilla_outputs = []
lemmatized_cons = []
infection_cons = []


def change_format(x):
    return x.replace('[','').replace(']','').replace("\"","").replace(', not','')

with open('./t5_input_w_mask.txt') as f:
    inputs = [x.replace('\n','') for x in f.readlines()]

with open('./GPT_output_w_m.txt') as f:
    gpt_outputs = [x.replace('\n','') for x in f.readlines()]

with open('./t5_3b_output_w_m.txt') as f:
    neuro_outputs = [x.replace('\n','') for x in f.readlines()]

with open('./t5_3b_output_vanilla_w_m.txt') as f:
    vanilla_outputs = [x.replace('\n','') for x in f.readlines()]

with open('./constraint_lemmas.json') as f:
    lemmatized_cons = [x.replace('\n','') for x in f.readlines()]
    lemmatized_cons = list(map(change_format,lemmatized_cons))

with open('./constraint_inflections.json') as f:
    infection_cons = [x.replace('\n','') for x in f.readlines()]


assert len(inputs) == len(gpt_outputs) == len(neuro_outputs) == len(vanilla_outputs) == len(lemmatized_cons) == len(infection_cons)



nl = '\n'
with open('./comparison_file.txt','w') as f:
    for index in range(len(inputs)):
        f.write(f'input_format:')
        f.write(nl)
        f.write(f'Input: {inputs[index]} ; Constraint: {lemmatized_cons[index]} ; Output:')
        f.write(nl)
        f.write(nl)
        f.write(f'These are constraints inflections used only for neuro algorithm')
        f.write(nl)
        f.write(infection_cons[index])
        f.write(nl)
        f.write(nl)
        f.write(nl)
        f.write(f'gpt : {gpt_outputs[index]}')
        f.write(nl)
        f.write(f'neuro : {neuro_outputs[index]}')
        f.write(nl)
        f.write(f'vanilla : {vanilla_outputs[index]}')
        f.write(nl)
        f.write(nl)
        f.write(nl)
        f.write('************************')
        f.write(nl)
        f.write(nl)