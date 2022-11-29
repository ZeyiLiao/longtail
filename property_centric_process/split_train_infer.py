from sklearn.model_selection import train_test_split
import jsonlines
import random
from pathlib import Path
import json
import csv
random.seed(42)

num_per_group = 2
train_group = 100

dir_o = '../longtail_data/raw_data/property_centric'
Path(dir_o).mkdir(exist_ok=True,parents=True)

all_data = []

with jsonlines.open('./all_data.jsonl') as f:
    for line in f:
        all_data.append(line)


group_index = random.sample(range(int(len(all_data)/num_per_group)),train_group)
group_index = sorted(group_index)
train_index = []

for _ in group_index:
    for i in range(num_per_group):
        train_index.append(_*num_per_group + i)

infer_index = list(set(range(len(all_data))).difference(train_index))

train_data = [all_data[_] for _ in train_index]

infer_data = [all_data[_] for _ in infer_index]

previous_index = -1
nl = '\n'

data_dict = {'train':train_data,'infer':infer_data}
for key in data_dict.keys():
    data = data_dict[key]
    id_l = []
    for line in data:
        id = line['id']
        id_l.append([id])

    with open(f'{dir_o}/{key}_ids.csv','w') as f:
        writer = csv.writer(f)
        writer.writerows(id_l)


            

    


