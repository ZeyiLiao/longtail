from tracemalloc import start
from helper import *
import requests
import time
import sys
from nli import NLI
from nltk.stem import WordNetLemmatizer
import torchtext
import torch.nn.functional as F
import pickle

lemmatizer = WordNetLemmatizer()

def dependency_parse(predictor,inputs):

    result_dict = dict()
    for input in inputs:
        dp = predictor.predict(sentence=input)
        words = dp['words']
        selected_indices = dp['pos']

        tmp =ddict(list)

        for index,_ in enumerate(selected_indices):
            if _ == 'NOUN':
                tmp['noun'].append(lemmatizer.lemmatize(words[index]))
            elif _ == 'VERB':
                tmp['verb'].append(lemmatizer.lemmatize(words[index]))

        result_dict[input] = tmp
    return result_dict

def filter_lemma(premise_words):

    for premise in premise_words.keys():

        words = premise_words[premise]
        lemma_filter = []
        tmp = []
        for word in words:
            if lemmatizer.lemmatize(word) not in lemma_filter:
                tmp.append(word)
            lemma_filter.append(lemmatizer.lemmatize(word))
            lemma_filter = list(set(lemma_filter))

        premise_words[premise] = tmp

    return premise_words

# def filter_hyper(premise_words):
#     for premise in premise_words.keys():
#         tmp = set()
#         satisfied_words = []
#         for word in premise_words[premise]:
#             word = word.replace(' ','_')
#             tmp_syn = set(wordnet.synsets(word))
#             # wordnet 找不到 我就认为是生僻词
#             if len(tmp_syn) == 0:
#                 continue

#             if len(tmp & tmp_syn) == 0:
#                 satisfied_words.append(word)
#                 tmp = tmp | tmp_syn
#         premise_words[premise] = satisfied_words

#     return premise_words



def filter_hyper(premise_words):
    for premise in premise_words.keys():
        word_synsets_dict = dict()
        word_synsets_dict_final = dict()

        for word in premise_words[premise]:
            word = word.replace(' ','_')
            tmp_syn = wordnet.synsets(word)
            # wordnet 找不到 我就认为是生僻词
            if len(tmp_syn) == 0:
                continue


            key_filter = []
            insert = True
            # dict 中每个word 都是互不包含的，但是有可能有交集。
            for key in word_synsets_dict.keys():
                synsets = word_synsets_dict[key]
                # if len(set(synsets) & set(tmp_syn)) != 0:
                if set(synsets).issubset(set(tmp_syn)) or set(tmp_syn).issubset(set(synsets)):

                    if len(tmp_syn) > len(synsets):
                        key_filter.append(key)
                    elif tmp_syn == synsets:
                        insert = False
                        break
                    elif len(tmp_syn) < len(synsets):
                        insert = False
                        break

            for key in key_filter:
                word_synsets_dict.pop(key)

            if insert:
                word_synsets_dict[word] = tmp_syn



        premise_words[premise] = list(word_synsets_dict.keys())

    return premise_words


def metric(l1,l2,sim_method):
    if sim_method == 'cos':
        return F.cosine_similarity(l1, l2)
    elif sim_method == 'distance':

        return torch.norm(l1-l2,dim=1)


def glove_based(query,words,glove,dim,device,sim_method):

    all_zero = torch.zeros(dim)
    query_emb = glove[query]
    query_emb = query_emb.expand(len(words),dim)

    words_emb = []
    bad_ids = []

    for index,word in enumerate(words):
        word_subs = word.split('_')
        tmp = all_zero
        for word_sub in word_subs:
            emb = glove[word_sub]
            if torch.equal(emb,all_zero):
                print("This word dosen't exist in glove")
                tmp = all_zero
                bad_ids.append(index)
                break
            tmp += emb
        tmp = tmp/len(word_subs)
        words_emb.append(tmp)

    words_emb = torch.stack(words_emb,dim = 0)
    assert words_emb.shape == query_emb.shape
    query_emb = query_emb.to(device)
    words_emb = words_emb.to(device)

    score = metric(query_emb,words_emb,sim_method).cpu().numpy()

    for id in bad_ids:
        score[id] -= 9999
    return score




def numbatch_based(query,words,numbatch,dim,device,sim_method):
    prepend = '/c/en/'

    all_zero = torch.zeros(dim)
    query_emb = torch.tensor(list(numbatch.loc[prepend + query]))
    query_emb = query_emb.expand(len(words),dim)

    words_emb = []
    bad_ids = []

    for index,word in enumerate(words):
        word_subs = word.split('_')
        tmp = all_zero
        for word_sub in word_subs:
            try:
                emb = torch.tensor(list(numbatch.loc[prepend + word_sub]))
            except:
                print("Doesn't have embedding at num batch")
                tmp = all_zero
                bad_ids.append(index)
                break
            tmp += emb
        tmp = tmp/len(word_subs)
        words_emb.append(tmp)

    words_emb = torch.stack(words_emb,dim = 0)
    assert words_emb.shape == query_emb.shape
    query_emb = query_emb.to(device)
    words_emb = words_emb.to(device)

    score = metric(query_emb,words_emb,sim_method).cpu().numpy()

    for id in bad_ids:
        score[id] -= 9999
    return score





def counter_based(query,words,counter_fitted,dim,device,sim_method):

    all_zero = torch.zeros(dim)
    query_emb = torch.tensor(list(counter_fitted[query]))
    query_emb = query_emb.expand(len(words),dim)

    words_emb = []
    bad_ids = []

    for index,word in enumerate(words):
        word_subs = word.split('_')
        tmp = all_zero
        for word_sub in word_subs:
            try:
                emb = torch.tensor(list(counter_fitted[word_sub]))
            except:
                print("Doesn't have embedding at counter-fitted")
                tmp = all_zero
                bad_ids.append(index)
                break
            tmp += emb
        tmp = tmp/len(word_subs)
        words_emb.append(tmp)

    words_emb = torch.stack(words_emb,dim = 0)
    assert words_emb.shape == query_emb.shape
    query_emb = query_emb.to(device)
    words_emb = words_emb.to(device)

    score = metric(query_emb,words_emb,sim_method).cpu().numpy()

    for id in bad_ids:
        score[id] -= 9999
    return score




def model_based(query,words,tokenizer,embedding,dim,device,sim_method):

    query_id = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query))).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = torch.sum(embedding(query_id).squeeze(0),dim=0)/len(query_id)
        query_emb = query_emb.expand(len(words),dim)

        words_emb = []

        for index,word in enumerate(words):
            word = word.replace('_',' ')

            word_id = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))).unsqueeze(0).to(device)
            word_emb = torch.sum(embedding(word_id).squeeze(0),dim=0)/len(word_id)
            words_emb.append(word_emb)

    words_emb = torch.stack(words_emb,dim = 0)
    assert words_emb.shape == query_emb.shape
    assert words_emb.device == query_emb.device

    score = metric(query_emb,words_emb,sim_method).cpu().numpy()

    return score






def calculate_sim(result_dict,premise_words,method,sim_method):
    premise_score = dict()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if method == 'glove':
        dim = 300
        glove = torchtext.vocab.GloVe(name="840B",dim=dim)

        for premise in premise_words.keys():
            query_words = []
            for key in list(result_dict[premise].keys()):
                query_words.extend(result_dict[premise][key])
            words = premise_words[premise]
            score_all = np.zeros(len(words))
            for query in query_words:
                score = glove_based(query,words,glove,dim,device,sim_method)
                score_all += score
            premise_score[premise] = score_all


    elif method == 'numbatch':
        dim = 300
        numbatch = pd.read_hdf('numbatch_embeddings/mini.h5')
        for premise in premise_words.keys():
            query_words = []
            for key in list(result_dict[premise].keys()):
                query_words.extend(result_dict[premise][key])
            words = premise_words[premise]
            score_all = np.zeros(len(words))
            for query in query_words:
                score = numbatch_based(query,words,numbatch,dim,device,sim_method)
                score_all += score
            premise_score[premise] = score_all


    elif method == 'counter_fitted':
        dim = 300
        with open('./counter_fit_embedding/counter_dict.pkl', 'rb') as f:
            counter_fitted = pickle.load(f)

        for premise in premise_words.keys():
            query_words = []
            for key in list(result_dict[premise].keys()):
                query_words.extend(result_dict[premise][key])
            words = premise_words[premise]
            score_all = np.zeros(len(words))
            for query in query_words:
                score = counter_based(query,words,counter_fitted,dim,device,sim_method)
                score_all += score
            premise_score[premise] = score_all



    elif method == 'model':
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        model = AutoModel.from_pretrained('roberta-large')
        embedding = model.embeddings
        dim = model.config.hidden_size
        embedding.to(device)
        # ids = tokenizer('I want to go',return_tensors='pt')
        # model(**ids)

        for premise in premise_words.keys():
            query_words = []
            for key in list(result_dict[premise].keys()):
                query_words.extend(result_dict[premise][key])
            words = premise_words[premise]
            score_all = np.zeros(len(words))
            for query in query_words:
                score = model_based(query,words,tokenizer,embedding,dim,device,sim_method)
                score_all += score
            premise_score[premise] = score_all


    return premise_score







def synonym(inputs):
    synonyms = []
    for input in inputs:
        for syn in wordnet.synsets(input):
            for l in syn.lemmas():
                synonyms.append(l.name())
    overlap = set(synonyms)&set(inputs)
    return list(set(synonyms) - overlap)






def nltk_based(result_dict):
    premise_words = dict()
    for premise in result_dict.keys():
        words_dict = result_dict[premise]
        candidates = []
        for pos in words_dict.keys():
            candidates += words_dict[pos]
        premise_words[premise] = synonym(candidates)
    return premise_words




def conceptnet_based(result_dict):
    prepend = 'http://api.conceptnet.io/c/en'
    premise_words = dict()
    for premise in result_dict.keys():

        words_dict = result_dict[premise]
        candidates = []
        for pos in words_dict.keys():
            candidates += words_dict[pos]
        premise_related_words = []

        for query in candidates:
            time.sleep(1)
            offset = 0
            limit = 1000
            url = f'{prepend}/{query}?offset={offset}&limit={limit}'

            # excluded_rel = ['ExternalURL','Synonym','Antonym','DistinctFrom','IsA','FormOf','EtymologicallyRelatedTo','EtymologicallyDerivedFrom']
            excluded_rel = ['ExternalURL']

            related_words = []

            obj = requests.get(f'{url}').json()
            edges = obj['edges']

            while (len(edges) != 0):

                for edge in tqdm(edges):
                    rel = edge['rel']['label']
                    score = edge['weight']
                    if rel not in excluded_rel:
                        # if score > 2:
                        #     continue
                        if edge['start']['language'] != 'en' or edge['end']['language'] != 'en':
                            continue
                        start = edge['start']['label'].lower()
                        end = edge['end']['label'].lower()
                        if query in start and query in end:
                            continue
                        else:
                            if query in start:
                                related_words.append(end)
                            elif query in end:
                                related_words.append(start)
                            else:
                                raise 'Sth wrong'
                premise_related_words += related_words


                offset += limit
                url = f'{prepend}/{query}?offset={offset}&limit={limit}'
                obj = requests.get(f'{url}').json()
                edges = obj['edges']


        premise_words[premise] = premise_related_words
    return premise_words


def global_glove(result_dict,sim_method,reverse):
    premise_words = ddict(list)
    top_k = 10000
    dim = 300
    glove = torchtext.vocab.GloVe(name="840B",dim=dim)

    for premise in result_dict.keys():
        query_words = []
        for key in list(result_dict[premise].keys()):
            query_words.extend(result_dict[premise][key])

        score_total = torch.zeros(glove.vectors.shape[0])

        for query in query_words:
            query = glove[query].expand(glove.vectors.shape[0],dim)
            score_tmp = metric(query,glove.vectors,sim_method)
            score_total += score_tmp

        dists = sorted(enumerate(score_total) , key = lambda i: i[1], reverse= reverse)

        for i in range(top_k):
            tmp = glove.itos[dists[i][0]]
            if (tmp not in query_words) and (tmp not in premise):
                premise_words[premise].append(tmp)

    return premise_words


def global_counter_fitting(result_dict,sim_method,reverse):
    premise_words = ddict(list)
    top_k = 10000
    dim = 300
    with open('./counter_fit_embedding/counter_dict.pkl', 'rb') as f:
        counter_fitted = pickle.load(f)
    all_embeddings = []
    all_words = []
    for word in counter_fitted.keys():
        all_words.append(word)
        all_embeddings.append(counter_fitted[word])

    all_embeddings = torch.tensor(np.vstack(all_embeddings))

    for premise in result_dict.keys():
        query_words = []
        for key in list(result_dict[premise].keys()):
            query_words.extend(result_dict[premise][key])

        score_total = torch.zeros(all_embeddings.shape[0])
        for query in query_words:
            query = torch.tensor(counter_fitted[query]).expand(all_embeddings.shape[0],dim)
            score_tmp = metric(query,all_embeddings,sim_method)
            score_total += score_tmp

        dists = sorted(enumerate(score_total) , key = lambda i: i[1], reverse= reverse)

        for i in range(top_k):
            tmp = all_words[dists[i][0]]
            if (tmp not in query_words) and (tmp not in premise):
                premise_words[premise].append(tmp)

    return premise_words




# glove,numbatch,model,counter_fitted
# cos, distance

# method_input,sim_input = 'counter_fitted','cos'
# use glove for global_glove
# use counter_fitted for global_counter_fitted
method_input,sim_input = sys.argv[1], sys.argv[2]


sim_method = sim_input
method = method_input

if sim_method == 'cos':
    reverse = True
elif sim_method == 'distance':
    reverse = False


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="8"



# sens = ["PersonX sneaks into PersonX's room","PersonX succeeds at speech","My flower have sunlights"]
sens = ["My flower sunlights"]

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
result_dict = dependency_parse(predictor,sens)
print(result_dict)


# premise_words = nltk_based(result_dict)

# premise_words = conceptnet_based(result_dict)

# premise_words = global_glove(result_dict,sim_method,reverse)

premise_words = global_counter_fitting(result_dict,sim_method,reverse)






premise_words = filter_lemma(premise_words)

premise_words = filter_hyper(premise_words)




premise_score = calculate_sim(result_dict,premise_words,method,sim_method)


for premise in premise_score.keys():
    score = premise_score[premise]
    sorted_id = sorted(range(len(score)), key=lambda k: score[k],reverse= reverse)
    premise_score[premise] = sorted_id

top_k = 30


print(f' Use {method} to get embedding and calcaute similarity by {sim_method}')
print(f'       ')
for premise in premise_score.keys():

    print(f'Premise is {premise}')
    print(f'top_k :{top_k} high simiar words are:')
    str = ''
    for id in premise_score[premise][:top_k]:
        str += ',' + premise_words[premise][id]
    print(str[1:])
    print('***********************************************************')
