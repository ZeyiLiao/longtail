import jsonlines
import json
import os



def process_t5_format(inputs,outputs,original_mask = '[mask]'):
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


def process_t5_format_continuation(inputs,outputs,original_mask = '[mask]'):

    new_output_l = []
    new_input_l = []
    for index in range(len(inputs)):
        input = inputs[index]
        output = outputs[index]
        prefix_end = input.index(original_mask)
        suffix_strat = prefix_end + len(original_mask)
        suffix_strat = len(input) - suffix_strat
        new_output = output[prefix_end:-suffix_strat]
        new_output_l.append(new_output)
        new_input_l.append(input.replace(original_mask,'<extra_id_0>'))

    return new_input_l,new_output_l
