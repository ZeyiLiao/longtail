from helper import *


def get_mask_logits(logits,column_indexs):
    ans = []
    if isinstance(column_indexs,list):
        ans = [logits[i,column_index].unsqueeze(0) for i,column_index in enumerate(column_indexs)]
        ans = torch.cat(ans,dim=0)
        return ans


class Decoder:
    def __init__(self,device):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

        self.model = AutoModelForMaskedLM.from_pretrained("microsoft/deberta-v3-large")
        self.device = torch.device(device)
        self.model.to(self.device)


    def __call__(self,composed_rules):
        decoder_result = ddict(list)
        for key in composed_rules.keys():
            sens = composed_rules[key]
            inputs = self.tokenizer(sens, truncation = True, padding = True,return_tensors="pt",max_length = 512)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                logits =self.model(**inputs).logits
                softmaxs = F.softmax(logits,dim = -1)
            # retrieve index of <mask>
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            row_indexs, column_indexs = mask_token_index
            column_indexs = column_indexs.numpy().tolist()
            decoder_result[key] = get_mask_logits(softmaxs,column_indexs)

        return decoder_result

    def KL_divergence(self,original_composed_rules_logits,composed_rules_logits):
        KL_result = ddict(lambda : ddict(list))
        for key in original_composed_rules_logits.keys():
            l1 = original_composed_rules_logits[key][0]
            l1 = torch.log(l1)
            for index,l2 in enumerate(composed_rules_logits[key]):
                result = F.kl_div(l1,l2).item()
                KL_result[key][index] = result

        return KL_result


    def jaccard(self,original_composed_rules_top_indices,composed_rules_top_indices):

        jaccard_result = ddict(lambda : ddict(list))
        for key in original_composed_rules_top_indices.keys():
            l1 = original_composed_rules_top_indices[key][0]
            for index,l2 in enumerate(composed_rules_top_indices[key]):
                intersection = len(list(set(l1).intersection(l2)))
                union = (len(l1) + len(l2)) - intersection
                result = float(intersection) / union
                jaccard_result[key][index] = result

        return jaccard_result



    def top_k_for_jaccard(self,composed_rules_softmaxs,top_k):
        composed_rules_top_indices = dict()
        for key in composed_rules_softmaxs.keys():
            logits = torch.topk(composed_rules_softmaxs[key],k=top_k,dim=-1)
            composed_rules_top_indices[key] = logits.indices.numpy().tolist()
        return composed_rules_top_indices