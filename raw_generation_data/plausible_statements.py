import json
import csv


with open('/home/zeyi/longtail/data/ATOMIC10X.json') as f:
    heads = []
    rels = []
    tails = []
    index = 0
    for line in f:
        data = json.loads(line)
        if data['p_valid_model'] > 0.99:
            heads.append(data['head'])
            rels.append(data['relation'])
            tails.append(data['tail'])

with open ('/home/zeyi/longtail/longtail_data/ATOMIC10X_filter.csv','w') as f:
    wrtier = csv.writer(f)
    for index,l in enumerate(zip(heads,rels,tails)):
        l = list(l)
        l.append(index)
        wrtier.writerow(l)