import pickle

with open('./all_objects_rank.pkl','rb') as f:
	data = pickle.load(f)

fo = open('./rank_by_relevance.txt','w')


nl = '\n'
for property in data.keys():
	fo.write(nl)
	fo.write(nl)
	fo.write('********')
	fo.write(nl)
	tmp = property
	fo.write(f'{property}')
	fo.write(nl)
	fo.write(nl)
	for method in data[tmp].keys():
		if method == 'concept':
			continue
		rank_list = data[tmp][method]
		rank_list = sorted(rank_list,key = lambda i : i[2],reverse=True)
		fo.write(nl)
		fo.write(f'Rank for {method}')
		fo.write(nl)
		fo.write(nl)

		for result in rank_list:
			fo.write(str(result))
			fo.write(nl)
			

