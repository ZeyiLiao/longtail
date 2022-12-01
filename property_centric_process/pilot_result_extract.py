import jsonlines


def filter_cons(cons):
    for con in cons:
        if '-' in con:
            return True
        if len(con.split(' ')) >= 3:
            return True
    return False

all_l = []
index = 1000
with jsonlines.open('./sentence_pilot_result_combined.jsonl') as f:
    for line in f:
        cons = line['constraint']
        if (len(cons['verb']) * len(cons['noun'])) == 0:
            continue
        if (filter_cons(cons['verb']) |  filter_cons(cons['noun'])):
            continue
        # remove the be words. Just check what I did in extract.py
        line['index'] = index
        all_l.append(line)
        index += 1

with jsonlines.open('./pilot_samples.jsonl','w') as f:
    f.write_all(all_l)