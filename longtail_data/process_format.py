import jsonlines
import json
import os



def process_format_wo_m_t5(inputs,outputs,original_mask = '[mask]'):
    new_output_l = []
    new_input_l = []
    for index in range(len(inputs)):
        input = inputs[index].replace(original_mask,'').strip()
        output = outputs[index].replace(input,'').strip()
        new_input_l.append(input)
        new_output_l.append(output)
    return new_input_l,new_output_l


def process_format_w_m_t5(inputs,outputs,original_mask = '[mask]'):
    new_output_l = []
    new_input_l = []
    for index in range(len(inputs)):
        input = inputs[index]
        output = outputs[index]
        prefix_end = input.index(original_mask)
        suffix_strat = prefix_end + len(original_mask)
        suffix_strat = len(input) - suffix_strat
        new_output = '<extra_id_0> ' + output[prefix_end:-suffix_strat] + ' <extra_id_1>'

        new_output_l.append(new_output)
        new_input_l.append(input.replace(original_mask,'<extra_id_0>'))
    return new_input_l,new_output_l


def process_format_wo_m_gpt2(inputs,outputs,original_mask = '[mask]'):
    new_output_l = []
    new_input_l = []
    for index in range(len(inputs)):
        output = outputs[index]
        new_input_l.append(output)
        new_output_l.append(output)
    return new_input_l,new_output_l