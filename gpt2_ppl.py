from helper import *
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTppl():
    def __init__(self,device = 'cpu'):

        self.model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj")
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")
        self.device = torch.device(device)
        self.model.to(self.device)

    def have_ppl_pair(self,text,text_pair,choice='normal',top_k = 5):
        length_text = len(text.split(' '))
        lenght_text_pair = [len(text.split(' ')) for text in text_pair]

        text_ppl_indexs = ddict(list)


        if choice == 'normal':
            sens = [f'{text} and {text_pair[i]},' for i in range(len(text_pair))]
            features = [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sen))) for sen in sens]

        elif choice == 'reverse':
            sens = [f'{text_pair[i]} and {text}' for i in range(len(text_pair))]
            features = [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sen))) for sen in sens]
        else:
            raise ("choice should be in ['normal','reverse']")

        with torch.no_grad():
            for index,feature in enumerate(features):
                feature = feature.to(self.device)
                loss = self.model(
                    feature,
                    labels = feature
                ).loss
                text_ppl_indexs[index] = round(math.exp(loss.item())/(length_text + lenght_text_pair[index]),2)

        text_ppl_indexs = sorted(text_ppl_indexs.items(), key= lambda item:item[1])[:top_k]
        text_ppl_indexs = {l[0]:l[1] for l in text_ppl_indexs}

        return text_ppl_indexs



    def have_ppl(self,texts,top_k = 20):
        features = [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sen))) for sen in texts]
        lenght_text_pair = [len(text.split(' ')) for text in texts]
        text_ppl_indexs = ddict(list)


        for index,feature in enumerate(features):
            feature = feature.to(self.device)
            loss = self.model(
                feature,
                labels = feature
            ).loss
            text_ppl_indexs[index] = round(math.exp(loss.item())/(lenght_text_pair[index]),2)

        text_ppl_indexs = sorted(text_ppl_indexs.items(), key= lambda item:item[1])[:top_k]
        text_ppl_indexs = {l[0]:l[1] for l in text_ppl_indexs}
        return text_ppl_indexs