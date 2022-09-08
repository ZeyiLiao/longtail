import sys

from retrieve import Retrieve
from helper import *
from nli import NLI

def filter_neutral(retrieve:Retrieve):
    pair_dict = ddict(lambda : ddict(list))
    constraints = []
    with open ('../output_file/constraints/constraint.json') as f:
        for line in f:
            _ = json.loads(line)
            constraints.append(_)

    decoded_result = []
    with open ('../output_file/constraints/decoded_result.txt') as f:
        for line in f:
            decoded_result.append(line.rstrip())

    with open ('../output_file/constraints/pt.txt') as f:
        for index,line in enumerate(f):
            # line = ' and'.join(line.rsplit(' and')[:-1])
            line = line.rstrip()

            pair_dict[line]['decoded'].append(decoded_result[index])
            pair_dict[line]['constraints'].append(constraints[index])


    for key in pair_dict.keys():
        query = key
        text_pair = pair_dict[key]['decoded']
        _, neutral_pair_index = retrieve.nli(query,text_pair)
        neutral_pair = [query + ' ' + text_pair[index] for index in neutral_pair_index]
        pair_dict[key]['decoded'] = neutral_pair
        tmp = pair_dict[key]['constraints']
        pair_dict[key]['constraints'] = [tmp[index] for index in neutral_pair_index]
    return pair_dict
