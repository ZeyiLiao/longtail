import pickle
import requests
from transformers import AutoTokenizer,AutoModel
from collections import defaultdict as ddict
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from tqdm import tqdm

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

def rank_by_bert(combinations):
	combinations_score = []

	
	for combination in combinations:
		id_0 = torch.tensor(t.convert_tokens_to_ids(t.tokenize(combination[0]))).unsqueeze(0).to(device)
		id_1 = torch.tensor(t.convert_tokens_to_ids(t.tokenize(combination[1]))).unsqueeze(0).to(device)
		embd_0 = sum_embeddings(id_0,embedding)
		embd_1 = sum_embeddings(id_1,embedding)
		sim_score = F.cosine_similarity(embd_1,embd_0,dim=0)
		sim_score = sim_score.cpu().detach().numpy().tolist()

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




with open('./all_objects_combination.pkl','rb') as f:
	properties = pickle.load(f)

m = AutoModel.from_pretrained('bert-large-uncased')
m.eval()
t = AutoTokenizer.from_pretrained('bert-large-uncased')
embedding = m.embeddings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
embedding.to(device)

properties_scores = ddict(lambda : ddict(list))
for key in tqdm(properties.keys()):
	combinations = properties[key]
	
	concept_scores = rank_by_conceptnet(combinations)
	sorted(concept_scores, key = lambda i : i[2],reverse=True)
	properties_scores[key]['concept'] = concept_scores

	concept_scores = rank_by_bert(combinations)
	sorted(concept_scores, key = lambda i : i[2],reverse=True)
	properties_scores[key]['bert'] = concept_scores

	concept_scores = rank_by_nltk(combinations)
	sorted(concept_scores, key = lambda i : i[2],reverse=True)
	properties_scores[key]['wordnet'] = concept_scores

properties_scores = dict(properties_scores)
with open('./all_objects_rank.pkl','wb') as f:
	pickle.dump(properties_scores,f)



		

			

			

