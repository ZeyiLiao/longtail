import pickle
import requests
from transformers import AutoTokenizer,AutoModel
from collections import defaultdict as ddict
import os
from argparse import ArgumentParser

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '9'

from numpy import dot
from numpy.linalg import norm
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import glob

import csv

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def rough_match(candicate,start_l):
	res = []
	for _ in candicate:
		res.append( _ in start_l)
	return all(res)

def score_from_conceptnet(combination):

	query = '_'.join(combination[0].split(' '))
	candidate = combination[1].split(' ')
	prepend = 'http://api.conceptnet.io/c/en'
	offset = 0
	limit = 1000
	url = f'{prepend}/{query}?offset={offset}&limit={limit}'
	obj = requests.get(f'{url}').json()

	# english_count
	eng_count = 0
	weight_all = 0
	for edge in obj['edges']:
		try:
			if edge['start']['language'] != 'en' or edge['end']['language'] != 'en':
				continue
			eng_count += 1
		except:
			continue
		
		start = edge['start']['label'].lower().split(' ')
		end = edge['end']['label'].lower()
		weight = edge['weight']
		
		if rough_match(start_l=start,candicate=candidate):
			weight_all += weight

	return weight_all/eng_count if eng_count != 0 else -10000
	


def rank_by_conceptnet(combinations):
	combinations_score = []
	
	for combination in combinations:
		score_all = 0
		score_all += score_from_conceptnet(combination)
		_combination = [combination[1],combination[0]]
		score_all += score_from_conceptnet(_combination)

		combination_score = combination + (score_all,)
		combinations_score.append(combination_score)

	return combinations_score

def sum_embeddings(word_id,embedding):
	return torch.sum(embedding(word_id).squeeze(0),dim=0)/len(word_id)

def rank_by_bert(combinations,words_to_vecs):
	combinations_score = []

	for combination in combinations:
		embd_0 = words_to_vecs[combination[0]]
		embd_1 = words_to_vecs[combination[1]]
		sim_score =dot(embd_0, embd_1)/(norm(embd_0)*norm(embd_1))


		combination_score = combination + (sim_score,)
		combinations_score.append(combination_score)

	return combinations_score
		

def score_by_nltk(combination):
	score = 0
	query = '_'.join(combination[0].split(' '))
	candidate = '_'.join(combination[1].split(' '))
	query_syns = wn.synsets(query)
	candidate_syns = wn.synsets(candidate)

	if len(query_syns) * len(candidate_syns) == 0:
		return -1000


	for query_syn in query_syns:
		for candidate_syn in candidate_syns:
			try:
				score += query_syn.path_similarity(candidate_syn)
			except:
				score += -1000
	return score/ (len(query_syns) * len(candidate_syns))
	
	


def rank_by_nltk(combinations):
	combinations_score = []
	
	for combination in combinations:
		score_all = 0
		score_all += score_by_nltk(combination)

		combination_score = combination + (score_all,)
		combinations_score.append(combination_score)

	return combinations_score



def filter_objects(combinations):
	long_l = []
	new_combinations = []
	for combination in combinations:
		if combination[2] >= 0.6:
			long_one = combination[0] if len(combination[0]) > len(combination[1]) else combination[1]
			long_l.append(long_one)

	combinations = list(filter(lambda i : len(set(i[:2]) & set(long_l))== 0, combinations))
	combinations = list(filter(lambda i : (i[2] <= 0.55 and i[2] >= 0.2), combinations))

	return combinations


	

def change_words_to_vec(all_objects,bs = 64):
	words_to_vecs = {}
	
	for index in range(0,len(all_objects),bs):
		objects_bs = all_objects[index:index+bs]
		ids = t(objects_bs,padding=True,return_tensors='pt',add_special_tokens=False)
		ids = ids.to(DEVICE)
		input_ids = ids['input_ids']
		attention_mask = ids['attention_mask']
		with torch.no_grad():
			all_vecs = embedding(input_ids)


		for i,word in enumerate(objects_bs):
			start_not_pad = torch.sum(attention_mask[i,:]).cpu().numpy()
			words_to_vecs[word] = (torch.sum(all_vecs[i,:start_not_pad,:],dim=0).cpu().numpy()/start_not_pad)

	return words_to_vecs


if __name__ == "__main__":
	parse = ArgumentParser()
	parse.add_argument('--vectors',action='store_true')
	args = parse.parse_args()

	if args.vectors:
		m = AutoModel.from_pretrained('bert-large-uncased')
		# m = AutoModel.from_pretrained('bert-base-uncased')

		t = AutoTokenizer.from_pretrained('bert-large-uncased')
		embedding = m.embeddings
		embedding.eval()

		# DEVICE = 'cpu'
		print(DEVICE)
		embedding.to(DEVICE)


	path = '../property_centric'
	properties_path = list(set(glob.glob(f'{path}/*')) - set(glob.glob(f'{path}/*.txt')))

	for property_path in tqdm(properties_path):
		all_objects_path = f"{property_path}/all_objects_combination.pkl"

		with open(all_objects_path,'rb') as f:
			combinations = pickle.load(f)
			
			# concept_scores = rank_by_conceptnet(combinations)
			# sorted(concept_scores, key = lambda i : i[2],reverse=True)
			# properties_scores[key]['concept'] = concept_scores

			# concept_scores = rank_by_nltk(combinations)
			# sorted(concept_scores, key = lambda i : i[2],reverse=True)
			# properties_scores[key]['wordnet'] = concept_scores

			if not os.path.exists(f"{property_path}/words_to_vecs.pkl"):
				assert args.vectors,'Need model to vectorlize words'

				all_objects = list(set([_[0] for _ in combinations]) | set([_[1] for _ in combinations]))
				words_to_vecs = change_words_to_vec(all_objects)

				with open(f"{property_path}/words_to_vecs.pkl",'wb') as f:
					pickle.dump(words_to_vecs,f)

			with open(f"{property_path}/words_to_vecs.pkl",'rb') as f:
				words_to_vecs = pickle.load(f)
				concept_scores = rank_by_bert(combinations,words_to_vecs)

			concept_scores = filter_objects(concept_scores)
			concept_scores = sorted(concept_scores, key = lambda i : i[2],reverse=True)
			with open(f"{property_path}/filtered_object_pairs.csv",'w') as f:
				writer = csv.writer(f)
				writer.writerows(concept_scores)
			




			

				

				

