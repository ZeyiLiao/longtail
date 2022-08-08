from helper import *

class NLI:
    def __init__(self,device = 'cpu'):

        self.model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-distilroberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-distilroberta-base')
        self.device = torch.device(device)
        self.model.to(self.device)

    def __call__(self,text,text_pair):
        text = [text] * len(text_pair)
        features = self.tokenizer(text, text_pair,  padding=True, truncation=True, return_tensors="pt")
        features = features.to(self.device)
        self.model.eval()

        with torch.no_grad():
            scores = self.model(**features).logits
            label_mapping = ['contradiction', 'entailment','neutral']
            labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
            index = self.select_neutral(labels,text_pair)
        return index

    def select_neutral(self,l,texts):
        ans = []
        for index,i in enumerate(l):
            if i == 'neutral':
                ans.append(texts[index])
        return ans