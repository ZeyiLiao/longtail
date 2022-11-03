import csv
from dataclasses import dataclass
import backoff
from gpt_body import *
import jsonlines


@dataclass
class PromptConfig_neg:
    engine: str = "text-davinci-002"
    max_tokens: int = 256
    temperature: float = 0.5
    top_p: float = 1
    logprobs: int = 0
    n: int = 1
    echo: bool = False

@dataclass
class PromptConfig_generation:
    engine: str = "text-davinci-002"
    max_tokens: int = 256
    temperature: float = 0.73
    top_p: float = 1
    logprobs: int = 0
    n: int = 1
    echo: bool = False



def conti_format(head,neg_tail):
    return f'Although {head}, {neg_tail[0] + neg_tail[1:-1]} because'


def tail_verbal(tail,rel):
    if rel == 'xAttr':
        stm = f'PersonX is seen as {tail}.'
        return stm
    if rel == 'xEffect':
        stm = f'PersonX {tail}.'
        return stm
    if rel == 'xIntent':
        stm = f'PersonX intends {tail}.'
        return stm
    if rel == 'xNeed':
        stm = f'PersonX has {tail}.'
        return stm
    if rel == 'xReact':
        stm = f'PersonX feels {tail}.'
        return stm
    if rel == 'xWant':
        stm = f'PersonX wants {tail}.'
        return stm
    else:
        raise NotImplementedError




def stm_verbal(head,tail,rel,insert_mask = False):
    tail = tail[0].lower() + tail[1:]
    if insert_mask:
        head = head + ' [mask]'
    if rel == 'xAttr':
        stm = f'{head}, so {tail}'
        return stm
    if rel == 'xEffect':
        stm = f'{head}, as a result, {tail}'
        return stm
    if rel == 'xIntent':
        stm = f'{head}, so {tail}'
        return stm
    if rel == 'xNeed':
        stm = f'Before {head[0].lower() + head[1:]}, {tail}'
        return stm
    if rel == 'xReact':
        stm = f'{head}, so {tail}'
        return stm
    if rel == 'xWant':
        stm = f'{head}, so {tail}'
        return stm
    else:
        raise NotImplementedError
    

def replace_mask(input ,replacement, mask = '[mask]'):
    replacement = 'and ' + replacement[:-1]
    input = input.replace(mask,replacement)
    return input
    


demon_negation = \
'Negate the statement:\n'\
"Original: PersonX is seen as adverturous.\n"\
"Negated: PersonX is not seen as adverturous.\n"\
"Original: PersonX dates someone new.\n"\
"Negated: PersonX does not date someone new.\n"\
"Original: PersonX intends to read the newspaper.\n"\
"Negated: PersonX does not intend to read the newspaper.\n"\
"Original: PersonX has to spend time with people\n"\
"Negated: PersonX does not have to spend time with people\n"\
"Original: PersonX feels loved.\n"\
"Negated: PersonX does not feel loved.\n"\
"Original: PersonX wants to take a shower.\n"\
"Negated: PersonX does not want to take a shower.\n"\
"Original: PersonX can’t find the phone to call the doctor.\n"\
"Negated: PersonX can find the phone to call the doctor."

demon_longtail = \
"Although personX leaves the burning building, personX does not intend to save her life because personX wants to go back to save her child.\n"\
"Although personX sneaks into room, personX does not feel nervous because personX is a professional thief."







# gpt_wrapper = PromptWrapper(PromptConfig_neg())

# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
# def gpt_generate(input,prefix, neg = False):
#     return gpt_wrapper.prompt_generation(input, prefix, neg)


# limit = 50
# index = -1
# save_l = []
# with open('/home/zeyi/longtail/longtail_data/ATOMIC10X_filter.csv') as f:
#     reader = csv.reader(f)
#     for line in reader:
#         index += 1
#         if index > limit:
#             break
#         head,rel,tail,id = line[0],line[1],line[2],line[3]
#         verbal_tail = tail_verbal(tail,rel)
#         neg_tail = gpt_generate(verbal_tail,prefix = demon_negation, neg = True)
        
#         conti_head = conti_format(head,neg_tail[0])

#         conti_generation = gpt_generate(conti_head, prefix = demon_longtail)

#         stm = stm_verbal(head,neg_tail[0],rel,insert_mask= True)
#         stm = replace_mask(stm,conti_generation[0])

#         save_dict = {'base': stm_verbal(head,verbal_tail,rel), 'expansion': conti_generation[0], 'base_expansion':stm  ,'neg_base': stm_verbal(head,neg_tail[0],rel), 'neg_base_mask': stm_verbal(head,neg_tail[0],rel,insert_mask= True) }
#         save_l.append(save_dict)


# with jsonlines.open('/home/zeyi/longtail/event_centric/data/data.jsonl','w') as f:
#     f.write_all(save_l)


fo = open('/home/zeyi/longtail/event_centric/data/data.txt', 'w')
nl = '\n'

with jsonlines.open('/home/zeyi/longtail/event_centric/data/data.jsonl') as f:
    for line in f:

        base = line['base']
        base_expansion = line['base_expansion']

        fo.write(base)
        fo.write(nl)
        fo.write(nl)
        fo.write(base_expansion)
        fo.write(nl)
        fo.write(nl)
        fo.write(nl)
        fo.write('*********************')
        fo.write(nl)
        fo.write(nl)





