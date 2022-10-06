inputs = []
gpt_outputs = []
neuro_outputs = []
vanilla_outputs = []


with open('./t5_input_w_mask.txt') as f:
    inputs = [x.replace('\n','') for x in f.readlines()]

with open('./GPT_output_w_m.txt') as f:
    gpt_outputs = [x.replace('\n','') for x in f.readlines()]

with open('./t5_3b_output_w_m.txt') as f:
    neuro_outputs = [x.replace('\n','') for x in f.readlines()]

with open('./t5_3b_output_vanilla_w_m.txt') as f:
    vanilla_outputs = [x.replace('\n','') for x in f.readlines()]

assert len(inputs) == len(gpt_outputs) == len(neuro_outputs) == len(vanilla_outputs)

nl = '\n'
with open('./comparison_file.txt','w') as f:
    for index in range(len(inputs)):
        f.write(f'input: {inputs[index]}')
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