
import torch
from transformers import AutoTokenizer,AutoModelForMaskedLM
from collections import defaultdict as ddict
import torch.nn.functional as F

def get_mask_logits(logits,column_indexs):
    ans = []
    ans = [logits[i,column_index].unsqueeze(0) for i,column_index in enumerate(column_indexs)]
    return ans

def KL_divergence(l1,l2):
    l1 = torch.log(l1)
    result = F.kl_div(l1,l2).item()
    # l2 = torch.log(l2)
    # result = F.kl_div(l1,l2,log_target=True).item()
    return result


def get_topk(mask_logits_l,top_k):
    max_prob_token = [torch.topk(mask_logit,top_k) for mask_logit in mask_logits_l]
    return max_prob_token



class Decoder:
    def __init__(self,model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model.to(self.device)


    def __call__(self,sens):
        
        
        inputs = self.tokenizer(sens, truncation = True, padding = True,return_tensors="pt",max_length = 512)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            logits =self.model(**inputs).logits
            softmaxs = F.softmax(logits,dim = -1)
        # retrieve index of <mask>
        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        row_indexs, column_indexs = mask_token_index
        mask_logits_l = get_mask_logits(softmaxs,column_indexs)
        return mask_logits_l

    def get_prob_of_word(self,word,logits):
        id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
        # TODO BART T5 can do not only one word mask
        assert len(id) == 1,'can only get prob for one token'
        return logits[0,id]



decoder = Decoder('bert-base-uncased')
mask_token = decoder.tokenizer.mask_token
sent = [f'I want to go {mask_token}',f'I pass the exam today so I feel {mask_token}']

# get the logits of the mask token
mask_logits_l = decoder(sent)

max_prob_k = get_topk(mask_logits_l,top_k = 5)

result = KL_divergence(mask_logits_l[0],mask_logits_l[1])

# check if need G or _ or sth, idk
prob_of_happy = decoder.get_prob_of_word('happy',mask_logits_l[1])
print(prob_of_happy)
print(max_prob_k[1])
decoder.tokenizer.batch_decode(max_prob_k[0])



