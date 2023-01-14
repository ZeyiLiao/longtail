from itertools import combinations 
import pickle
import glob
import os
# path = './all_objects.pkl'
# with open(path,'rb') as f:
# 	_dict = pickle.load(f)
# 	for key in _dict.keys():
# 		objects = _dict[key]
# 		combines = combinations(objects,2)
# 		_dict[key] = list(combines)


path = '../property_centric'

properties_path = list(set(glob.glob(f'{path}/*')) - set(glob.glob(f'{path}/*.txt')))

for path in properties_path:
	verified_objects_path = f"{path}/objects_verified.txt"
	with open(verified_objects_path) as f:
		all_objects = [_.strip() for _ in f.readlines()]
		combines = list(combinations(all_objects,2))

	with open(f'{path}/all_objects_combination.pkl','wb') as f:
		pickle.dump(combines,f)