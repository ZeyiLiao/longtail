from helper import *
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTppl():
    def __init__(self,device = 'cpu'):

        self.model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.device = torch.device(device)
        self.model.to(self.device)

    def have_ppl_pair(self,text,text_pair,combine_order='normal',top_k = 5):
        length_text = len(text.split(' '))
        lenght_text_pair = [len(text.split(' ')) for text in text_pair]

        selected_text_ppl = ddict(list)


        if combine_order == 'normal':
            sens = [f'{text} and {text_pair[i]},' for i in range(len(text_pair))]
            features = [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sen))) for sen in sens]

        elif combine_order == 'reverse':
            sens = [f'{text_pair[i]} and {text}' for i in range(len(text_pair))]
            features = [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sen))) for sen in sens]
        else:
            raise ("combine_order should be in ['normal','reverse']")

        with torch.no_grad():
            for index,feature in enumerate(features):
                feature = feature.to(self.device)
                loss = self.model(
                    feature,
                    labels = feature
                ).loss
                # selected_text_ppl[text_pair[index]] = round(math.exp(loss.item())/(length_text + lenght_text_pair[index]),2)
                selected_text_ppl[text_pair[index]] = math.exp(loss.item())
        # note: here we want to select high ppl which means the sentence is less plausible
        selected_text_ppl = sorted(selected_text_ppl.items(), key= lambda item:item[1],reverse=True)[:top_k]
        selected_text_ppl = [l[0] for l in selected_text_ppl]

        return selected_text_ppl



    def have_ppl(self,composed_rules,top_k_ratio):
        composed_rules = list(set(composed_rules))
        features = [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(composed_rule))) for composed_rule in composed_rules]
        lenght_text_pair = [len(composed_rule.split(' ')) for composed_rule in composed_rules]
        selected_text_ppl_low = ddict(list)

        self.model.eval()

        with torch.no_grad():
            for index,feature in enumerate(features):
                feature = feature.to(self.device)
                loss = self.model(
                    feature,
                    labels = feature
                ).loss
                selected_text_ppl_low[composed_rules[index]] = round(math.exp(loss.item())/(lenght_text_pair[index]),2)

        # This is for probing the gpt-j, so we sould select higher liklihood which is low perplexity
        selected_text_ppl_low = sorted(selected_text_ppl_low.items(), key= lambda item:item[1])
        selected_text_ppl_low = selected_text_ppl_low[:int(len(composed_rules) * top_k_ratio)]
        selected_text_ppl_low = {l[0]:l[1] for l in selected_text_ppl_low}
        return selected_text_ppl_low