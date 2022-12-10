from itertools import combinations 
import pickle

path = './all_objects.pkl'
with open(path,'rb') as f:
	_dict = pickle.load(f)
	for key in _dict.keys():
		objects = _dict[key]
		combines = combinations(objects,2)
		_dict[key] = list(combines)

with open('./all_objects_combination.pkl','wb') as f:
	pickle.dump(_dict,f)