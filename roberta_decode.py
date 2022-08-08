from helper import *



def get_mask_logtis(logits,column_indexs):
    ans = []
    column_indexs = column_indexs.numpy().tolist()
    ans = [logits[i,column_index].unsqueeze(0) for i,column_index in enumerate(column_indexs)]
    ans = torch.cat(ans,dim=0)
    return ans

class Decoder:
    def __init__(self,device):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForMaskedLM.from_pretrained("roberta-base")
        self.device = torch.device(device)
        self.model.to(self.device)


    def decode(self,sens):
        inputs = self.tokenizer(sens, truncation = True, padding = True,return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            logits =self.model(**inputs).logits
        # retrieve index of <mask>
        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        row_index, column_index = mask_token_index
        ans_ids = get_mask_logtis(logits,column_index).argmax(axis = -1)

        ans = []
        for ans_id in ans_ids:
            ans.append(self.tokenizer.decode(ans_id))

        return ans