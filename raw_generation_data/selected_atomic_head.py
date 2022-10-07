import csv
from collections import defaultdict as ddict
import numpy as np
import random


path = '../data/ATOMIC10X_filter.csv'
tmp = ddict(list)
with open(path) as f:
    reader = csv.reader(f)
    for line in reader:
        head,rel,tail = line[0],line[1],line[2]
        tmp[head].append(rel)

desired_heads = []
for head in tmp.keys():
    if 'xReact' in tmp[head] and 'xAttr' in tmp[head]:
        desired_heads.append(head)


desired_heads = random.sample(desired_heads,300)
with open('../longtail_data/input.csv','w') as f:
    writer = csv.writer(f)
    for head in desired_heads:
        writer.writerow([head])