from tracemalloc import start
from helper import *
import requests
import time
import sys
import spacy
from nli import NLI
from nltk.stem import WordNetLemmatizer
import torchtext
import torch.nn.functional as F
import pickle
from nltk.corpus import wordnet


lemmatizer = WordNetLemmatizer()

nlp = spacy.load('en_core_web_sm')

def pos_keyword_extraction(inputs):
    #assert sent_path.endswith(".txt")
    for sent in inputs:
        doc = nlp(str(sent))
        keywords = []
        for token in doc:
            if (token.pos_.startswith('V') or token.pos_.startswith('PROP')) and token.is_alpha and not token.is_stop:
                keywords.append(token.lemma_)
        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                keywords.append(root_noun.lemma_)
    return keywords

def dependency_parse(predictor,inputs):

    premise_extraction = dict()
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

        premise_extraction[input] = tmp
    return premise_extraction


def filter_lemma_premise(premise_words,premise_extraction):

    for premise in premise_words.keys():
        query_words = []
        for key in list(premise_extraction[premise].keys()):
            query_words.extend(premise_extraction[premise][key])

        lemma_filter = []

        for word in query_words:
            lemma_filter.append(lemmatizer.lemmatize(word))

        tmp = []
        for word in premise_words[premise]:

            if lemmatizer.lemmatize(word) not in lemma_filter:
                tmp.append(word)
        premise_words[premise] = tmp

    return premise_words



def filter_syn_premise(premise_words,premise_extraction):

    for premise in premise_words.keys():
        query_words = []
        for key in list(premise_extraction[premise].keys()):
            query_words.extend(premise_extraction[premise][key])

        syn_filter = synonym(query_words)

        tmp = []
        for word in premise_words[premise]:

            if lemmatizer.lemmatize(word) not in syn_filter and word not in syn_filter:
                tmp.append(word)
        premise_words[premise] = tmp


    return premise_words


def filter_hyper_premise(premise_words,premise_extraction):

    for premise in premise_words.keys():
        query_words = []
        for key in list(premise_extraction[premise].keys()):
            query_words.extend(premise_extraction[premise][key])

        hyper_filter = []
        for query in query_words:
            hyper_filter.extend(wordnet.synsets(query))
        hyper_filter = set(hyper_filter)

        tmp = []
        for word in premise_words[premise]:

            if len(set(wordnet.synsets(word)) & hyper_filter) == 0 :
                tmp.append(word)
            else:
                pass
        premise_words[premise] = tmp


    return premise_words

def filter_noun_verb(premise_words):

    for premise in premise_words.keys():

        words = nltk.pos_tag(premise_words[premise])
        tmp = [word[0] for word in words if word[1][0] in ['N','V']]
        premise_words[premise] = tmp
    
    return premise_words

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


def metric(l1,l2,similarity_method):
    if similarity_method == 'cos':
        return F.cosine_similarity(l1, l2)
    elif similarity_method == 'distance':

        return torch.norm(l1-l2,dim=1)


def glove_based(query,words,glove,dim,device,similarity_method):

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

    score = metric(query_emb,words_emb,similarity_method).cpu().numpy()

    for id in bad_ids:
        score[id] -= 9999
    return score




def numbatch_based(query,words,numbatch,dim,device,similarity_method):
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

    score = metric(query_emb,words_emb,similarity_method).cpu().numpy()

    for id in bad_ids:
        score[id] -= 9999
    return score





def counter_based(query,words,counter_fitted,dim,device,similarity_method):

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

    score = metric(query_emb,words_emb,similarity_method).cpu().numpy()

    for id in bad_ids:
        score[id] -= 9999
    return score




def model_based(query,words,tokenizer,embedding,dim,device,similarity_method):
    embedding.to(device)
    with torch.no_grad():
        query_id = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query))).unsqueeze(0).to(device)

        query_emb = torch.sum(embedding(query_id).squeeze(0),dim=0)/len(query_id)
        query_emb = query_emb.expand(len(words),dim)

        words_emb = []

        for word in words:
            word = word.replace('_',' ')

            word_id = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))).unsqueeze(0).to(device)
            word_emb = torch.sum(embedding(word_id).squeeze(0),dim=0)/len(word_id)
            words_emb.append(word_emb)


    words_emb = torch.stack(words_emb,dim = 0)
    assert words_emb.shape == query_emb.shape
    assert words_emb.device == query_emb.device

    score = metric(query_emb,words_emb,similarity_method).cpu().numpy()

    return score


def union_related(top_k_union,similarity_method,premise_score,premise_words):



    reverse = reverse_state(similarity_method)
    for premise in premise_score.keys():
        scores = premise_score[premise]

        tmp = set(range(len(scores[0])))

        for score in scores:
            sorted_id = set(sorted(range(len(score)), key=lambda k: score[k],reverse= reverse)[:top_k_union])
            tmp = tmp & sorted_id

        premise_score[premise] = list(tmp)

    def retrieve_word(id):
        return premise_words[premise][id]

    for premise in premise_score.keys():
        premise_words[premise] = list(map(retrieve_word,premise_score[premise]))

    return premise_words




def calculate_sim(premise_extraction,premise_words,embedding_source,similarity_method):
    premise_score = dict()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if embedding_source == 'glove':
        dim = 300
        glove = torchtext.vocab.GloVe(name="840B",dim=dim)

        for premise in premise_words.keys():
            query_words = []
            for key in list(premise_extraction[premise].keys()):
                query_words.extend(premise_extraction[premise][key])
            words = premise_words[premise]
            score_all = np.zeros(len(words))

            tmp = []

            for query in query_words:
                score = glove_based(query,words,glove,dim,device,similarity_method)
                score_all += score
                tmp.append(score)
            premise_score[premise] = tmp


    elif embedding_source == 'numbatch':
        dim = 300
        numbatch = pd.read_hdf('numbatch_embeddings/mini.h5')
        for premise in premise_words.keys():
            query_words = []
            for key in list(premise_extraction[premise].keys()):
                query_words.extend(premise_extraction[premise][key])
            words = premise_words[premise]
            score_all = np.zeros(len(words))

            tmp = []

            for query in query_words:
                score = numbatch_based(query,words,numbatch,dim,device,similarity_method)
                tmp.append(score)
                score_all += score
            premise_score[premise] = tmp


    elif embedding_source == 'counter_fitted':
        dim = 300
        with open('./counter_fit_embedding/counter_dict.pkl', 'rb') as f:
            counter_fitted = pickle.load(f)

        for premise in premise_words.keys():
            query_words = []
            for key in list(premise_extraction[premise].keys()):
                query_words.extend(premise_extraction[premise][key])
            words = premise_words[premise]
            score_all = np.zeros(len(words))

            tmp = []

            for query in query_words:
                score = counter_based(query,words,counter_fitted,dim,device,similarity_method)
                tmp.append(score)
                score_all += score
            premise_score[premise] = tmp



    elif embedding_source == 'model':
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        model = AutoModel.from_pretrained('roberta-large')
        embedding = model.embeddings
        dim = model.config.hidden_size


        for premise in premise_words.keys():
            query_words = []
            for key in list(premise_extraction[premise].keys()):
                query_words.extend(premise_extraction[premise][key])
            words = premise_words[premise]
            score_all = np.zeros(len(words))

            tmp = []

            for query in query_words:
                score = model_based(query,words,tokenizer,embedding,dim,device,similarity_method)
                tmp.append(score)
                score_all += score
            premise_score[premise] = tmp


    return premise_score


def synonym(inputs):
    synonyms = []
    for input in inputs:
        for syn in wordnet.synsets(input):
            for l in syn.lemmas():
                synonyms.append(l.name())

    # overlap = set(synonyms)&set(inputs)

    return list(set(synonyms) | set(inputs))






def nltk_based(premise_extraction):
    premise_words = dict()
    for premise in premise_extraction.keys():
        words_dict = premise_extraction[premise]
        candidates = []
        for pos in words_dict.keys():
            candidates += words_dict[pos]
        premise_words[premise] = synonym(candidates)
    return premise_words




def conceptnet_based(premise_extraction):
    prepend = 'http://api.conceptnet.io/c/en'
    premise_words = dict()
    for premise in premise_extraction.keys():

        words_dict = premise_extraction[premise]
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
                    if rel not in excluded_rel:

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


def global_glove(premise_extraction,similarity_method):
    reverse = reverse_state(similarity_method)
    premise_words = ddict(list)
    top_k = 10000
    dim = 300
    glove = torchtext.vocab.GloVe(name="840B",dim=dim)

    for premise in premise_extraction.keys():
        query_words = []
        for key in list(premise_extraction[premise].keys()):
            query_words.extend(premise_extraction[premise][key])

        score_total = torch.zeros(glove.vectors.shape[0])

        for query in query_words:
            query = glove[query].expand(glove.vectors.shape[0],dim)
            score_tmp = metric(query,glove.vectors,similarity_method)
            score_total += score_tmp

        dists = sorted(range(len(score_total)) , key = lambda i: score_total[i], reverse= reverse)

        for i in range(top_k):
            tmp = glove.itos[dists[i]]
            if (tmp not in query_words) and (tmp not in premise):
                premise_words[premise].append(tmp)

    return premise_words


def global_counter_fitted(premise_extraction,similarity_method):
    reverse = reverse_state(similarity_method)
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

    for premise in premise_extraction.keys():
        query_words = []
        for key in list(premise_extraction[premise].keys()):
            query_words.extend(premise_extraction[premise][key])

        score_total = torch.zeros(all_embeddings.shape[0])
        for query in query_words:
            query = torch.tensor(counter_fitted[query]).expand(all_embeddings.shape[0],dim)
            score_tmp = metric(query,all_embeddings,similarity_method)
            score_total += score_tmp

        dists = sorted(range(len(score_total)) , key = lambda i: score_total[i], reverse= reverse)

        for i in range(top_k):
            tmp = all_words[dists[i]]
            if (tmp not in query_words) and (tmp not in premise):
                premise_words[premise].append(tmp)

    return premise_words


def reverse_state(similarity_method):
    if similarity_method == 'cos':
        reverse = True
    elif similarity_method == 'distance':
        reverse = False
    return reverse

def get_candidates(candidate_method,similarity_method,premise_extraction):
    if candidate_method == 'nltk':
        premise_words = nltk_based(premise_extraction)
    elif candidate_method == 'conceptnet':
        premise_words = conceptnet_based(premise_extraction)
    elif candidate_method == 'global_glove':
        premise_words = global_glove(premise_extraction,similarity_method)
    elif candidate_method == 'global_counter_fitted':
        premise_words = global_counter_fitted(premise_extraction,similarity_method)

    return premise_words

def filter_process(premise_words,premise_extraction):

    premise_words = filter_lemma(premise_words)

    premise_words = filter_lemma_premise(premise_words,premise_extraction)

    premise_words = filter_syn_premise(premise_words,premise_extraction)

    premise_words = filter_hyper_premise(premise_words,premise_extraction)

    premise_words = filter_hyper(premise_words)


    return premise_words




def main():


    # candidate_method :[nltk,conceptnet,global_glove,global_counter_fitted]
    # embedding_source: [glove,numbatch,model,counter_fitted]
    # similarity_method: [cos,distance]
    # candidate_method,embedding_source,similarity_method = sys.argv[1], sys.argv[2], sys.argv[3]
    candidate_method,embedding_source,similarity_method = 'conceptnet', 'model', 'cos'
    if candidate_method == 'global_glove':
        embedding_source = 'glove'
    elif candidate_method == 'global_counter_fitted':
        embedding_source = 'counter_fitted'




    # sens = ["PersonX sneaks into PersonX's room","PersonX succeeds at speech","My flower have sunlights"]
    sens = ["My flowers sunlight","PersonX sneaks into room"]

    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
    premise_extraction = dependency_parse(predictor,sens)
    print(premise_extraction)


    premise_words = get_candidates(candidate_method,similarity_method,premise_extraction)
    premise_words = filter_noun_verb(premise_words)



    premise_score = calculate_sim(premise_extraction,premise_words,embedding_source,similarity_method)

    top_k_union = 400
    premise_words = union_related(top_k_union,similarity_method,premise_score,premise_words)

    premise_words = filter_process(premise_words,premise_extraction)

    premise_score = calculate_sim(premise_extraction,premise_words,embedding_source,similarity_method)




    top_k = 50
    for premise in premise_score.keys():
        score_total = np.zeros(len(premise_score[premise][0]))
        for score in premise_score[premise]:
            score_total += np.array(score)
        sorted_id = sorted(range(len(score_total)),key = lambda i: score_total[i], reverse= reverse_state(similarity_method))[:top_k]
        premise_score[premise] = sorted_id


    for premise in premise_words:
        tmp = []
        sorted_id = premise_score[premise]
        for id in sorted_id:
            tmp.append(premise_words[premise][id])
        print(f'premise : {premise}')
        print(f'related words: {tmp}')

if __name__ == '__main__':
    main()
