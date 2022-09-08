from helper import *

class NLI:
    def __init__(self,device):
        self.label_mapping = {'roberta-large-mnli':['contradiction', 'neutral','entailment'],'cross-encoder/nli-distilroberta-base':['contradiction', 'entailment', 'neutral']}
        self.model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
        self.device = torch.device(device)
        self.model.to(self.device)



    def __call__(self,query,text_pair_ori):
        if query[-1] != '.':
            query += '.'
        tmp = []
        for text in text_pair_ori:
            if text[-1] != '.':
                tmp.append(text + '.')
            else:
                tmp.append(text)
        text_pair = tmp
        query = [query] * len(text_pair)
        features = self.tokenizer(query, text_pair,  padding=True, truncation=True, return_tensors="pt")
        features_reverse = self.tokenizer(text_pair, query,  padding=True, truncation=True, return_tensors="pt")
        features = features.to(self.device)
        features_reverse = features_reverse.to(self.device)


        self.model.eval()
        with torch.no_grad():
            scores = self.model(**features).logits
            label_mapping = self.label_mapping['roberta-large-mnli']
            labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
            neutral_texts,neutral_texts_index = self.select_neutral(labels,text_pair_ori)

            scores_reverse = self.model(**features_reverse).logits
            label_mapping = self.label_mapping['roberta-large-mnli']
            labels_reverse = [label_mapping[score_max] for score_max in scores_reverse.argmax(dim=1)]
            neutral_texts_reverse,neutral_texts_reverse_index = self.select_neutral(labels_reverse,text_pair_ori)


        return list(set(neutral_texts) & set(neutral_texts_reverse)),list(set(neutral_texts_index) & set(neutral_texts_reverse_index))



    def select_neutral(self,l,texts):
        ans = []
        ans_index = []
        for index,i in enumerate(l):
            if i == 'entailment':
                ans.append(texts[index])
                ans_index.append(index)
        return ans,ans_index
