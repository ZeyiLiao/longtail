from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV
import jsonlines
import random
from pathlib import Path
import json
import csv

random.seed(37)

num_per_group = 2

dir_o = '../longtail_data/raw_data/property_centric_test'
Path(dir_o).mkdir(exist_ok=True,parents=True)

all_data = []
train_group = 100
with jsonlines.open('./samples_process.jsonl') as f:
    for line in f:
        all_data.append(line)



group_index = random.sample(range(int(len(all_data)/num_per_group)),train_group)
group_index = sorted(group_index)
train_index = []

for _ in group_index:
    for i in range(num_per_group):
        train_index.append(_*num_per_group + i)

infer_index = list(set(set(range(len(all_data)))).difference(train_index))

train_data = [all_data[_] for _ in train_index]
infer_data = [all_data[_] for _ in infer_index]

previous_index = -1
nl = '\n'
data_dict = {'train':train_data,'infer':infer_data}
for key in data_dict.keys():
    data = data_dict[key]
    inflections_o = f'{dir_o}/inflections_{key}.json'
    inflections_o_l = []

    lemmas_o = f'{dir_o}/lemmas_{key}.json'
    lemmas_o_l = []

    stms_o = f'{dir_o}/stms_{key}.csv'
    stms_o_l = []

    for line in data:
        constraints_lemma = line['cons_lemma']
        id = line['id']
        index = line['index']

        stm = line['conte_template']
        lemmas_o_l.append(constraints_lemma)
        stms_o_l.append([stm,id])
        
        
        constraints_inflection = []
        for clause in constraints_lemma:
            tmp = []
            for word in clause:

                word_inflections = getAllInflections(word)
                if not word_inflections or len(word_inflections) == 0:
                    word_inflections = dict(getAllInflectionsOOV(word,'VERB'), **getAllInflectionsOOV(word,'NOUN'))
                    if len(word.split(' ')) == 1:
                        word_inflections.update(getAllInflectionsOOV(word,'ADJ'))
                tmp.extend(list(set([_[0] for _ in list(word_inflections.values())])))
            constraints_inflection.append(tmp)

        if 'neg' in id:
            constraints_inflection.append(['no', 'not'])

        inflections_o_l.append(constraints_inflection)

    with open(inflections_o,'w') as f:
        for line in inflections_o_l:
            json.dump(line,f)
            f.write(nl)

    with open(lemmas_o,'w') as f:
        for line in lemmas_o_l:
            json.dump(line,f)
            f.write(nl)

    with open(stms_o,'w') as f:
        writer = csv.writer(f)
        for line in stms_o_l:
            writer.writerow(line)
            

    


