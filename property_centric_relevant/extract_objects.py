import pickle
from collections import defaultdict as ddict
import jsonlines

root = '/home/zeyi/longtail/property_centric'
path = f'{root}/properties.txt'
properties = ddict(list)
with open(path) as f:
	for line in f:
		property = line.rstrip()
		decrease_p = f'{root}/{property}/decrease_1.jsonl'
		increase_p = f'{root}/{property}/increase_1.jsonl'

		with jsonlines.open(decrease_p) as f:
			for d in f:
				object = [d['object1'],d['object2']]
				properties[property].extend(object)

		properties[property] = list(set(properties[property]))

with open('./all_objects.pkl','wb') as f:
	pickle.dump(properties,f)

