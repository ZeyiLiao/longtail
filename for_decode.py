from helper import *


def get_mask_logits(logits,column_indexs):
    ans = []

    ans = [logits[i,column_index].unsqueeze(0) for i,column_index in enumerate(column_indexs)]
    ans = torch.cat(ans,dim=0)
    return ans


class Decoder:
    def __init__(self,device,model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)


    def __call__(self,composed_rules):
        decoder_result = ddict(lambda :ddict(list))
        for key in composed_rules.keys():
            for key2 in composed_rules[key].keys():
                sens = composed_rules[key][key2]
                inputs = self.tokenizer(sens, truncation = True, padding = True,return_tensors="pt",max_length = 512)
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    logits =self.model(**inputs).logits
                    softmaxs = F.softmax(logits,dim = -1)
                # retrieve index of <mask>
                mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
                row_indexs, column_indexs = mask_token_index

                decoder_result[key][key2] = get_mask_logits(softmaxs,column_indexs)

        return decoder_result


    def KL_divergence(self,original_composed_rules_logits,composed_rules_logits):
        KL_result = ddict(lambda : ddict( lambda : ddict(list)))
        for key in original_composed_rules_logits.keys():
            for key2 in original_composed_rules_logits[key]:
                l1 = original_composed_rules_logits[key][key2][0]
                l1 = torch.log(l1)
                for index,l2 in enumerate(composed_rules_logits[key][key2]):
                    l2 = torch.log(l2)
                    result = F.kl_div(l2,l1,log_target=True).item()
                    KL_result[key][key2][index] = result

        return KL_result


    def jaccard(self,original_composed_rules_top_indices,composed_rules_top_indices):

        jaccard_result = ddict(lambda : ddict(list))
        for key in original_composed_rules_top_indices.keys():
            l1 = original_composed_rules_top_indices[key][0]
            l1 = l1.cpu().numpy().tolist()
            for index,l2 in enumerate(composed_rules_top_indices[key]):
                l2 = l2.cpu().numpy().tolist()
                intersection = len(list(set(l1).intersection(l2)))
                union = (len(l1) + len(l2)) - intersection
                result = float(intersection) / union
                jaccard_result[key][index] = result

        return jaccard_result



    def top_k_for_jaccard(self,composed_rules_softmaxs,top_k):
        composed_rules_top_indices = dict()
        for key in composed_rules_softmaxs.keys():
            logits = torch.topk(composed_rules_softmaxs[key],k=top_k,dim=-1)
            composed_rules_top_indices[key] = logits.indices
        return composed_rules_top_indices


    def decode_to_word(self,indices):
        words = ddict(list)
        for key in indices.keys():
            tmp = indices[key]
            words[key] = self.tokenizer.batch_decode(indices[key])
        return words

    def mask_word_likelihood(self,softmax):
        mask_word_likelihood_dict = ddict(lambda :ddict(list))

        for key in softmax.keys():
            for key2 in softmax[key].keys():
                masked_word = ' ' + key2
                masked_word_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(masked_word))
                logtis = softmax[key][key2]
                mask_word_likelihood_dict[key][key2] = logtis[:,masked_word_id]
        return mask_word_likelihood_dict
