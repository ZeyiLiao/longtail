from helper import *
from nli import NLI
from gpt2_ppl import GPTppl
from deberta_decode import Decoder



def back_to_sen(head,relation,tail):
    if relation == 'xAttr':
        content = head + ', So PersonX is seen as ' + tail
    elif relation == 'xReact':
        content = head + ', So PersonX feels ' + tail
    elif relation == 'xNeed':
        content = 'Before ' + head + ', PersonX has ' + tail
    elif relation == 'xWant':
        content = head + ', So PersonX wants ' + tail
    elif relation == 'xIntent':
        content = head + ', Because PersonX intents ' + tail
    elif relation == 'xEffect':
        content = head + ', As a result, PersonX ' + tail
    elif relation == 'HinderedBy':
        content =  head + ', This is hindered if ' + tail
    return content




class Retrieve():
    def __init__(self,file_path,save_embedding_path,all_tuples_file,device):
        self.device = device
        self.embedder = SimCSE('princeton-nlp/sup-simcse-bert-base-uncased',device = device)
        # embedder.build_index(file_path,device=device,batch_size=64,save_path=save_embedding_path)
        self.embedder.load_embeddings(file_path,save_embedding_path)
        self.nli = NLI(device)
        self.gptppl = GPTppl(device)
        self.decoder = Decoder(device)
        self.df = pd.read_csv(all_tuples_file,names = ['head','relation','tail'],index_col='head')

    def query_neutral(self,query,top_k ,threshold ):
        results = self.embedder.search(query,device = self.device, top_k=top_k, threshold=threshold)

        text_pair = [text[0] for text in results]
        neutral_text_pair = self.nli(query,text_pair)
        return neutral_text_pair



    def select_highppl_neutral(self,query,neutral_text_pair,top_k,combine_order):
        selected_neutral_text_pair = self.gptppl.have_ppl_pair(query,neutral_text_pair,combine_order = combine_order,top_k = top_k)
        return selected_neutral_text_pair

    def query_relation_tail_ppl(self,query,composed_p):
        df_selected = self.df.loc[query]
        relations = list(df_selected['relation'])
        tails = list(df_selected['tail'])


        composed_rules = []
        for text_pair in composed_p:
            for index,relation in enumerate(relations):
                tail = tails[index]
                composed_rule = back_to_sen(text_pair,relation,tail)
                composed_rules.append(composed_rule)

        return composed_rules

    def query_relation_tail_mask(self,query,composed_p = None ,keep_attr_react = False):
        df_selected = self.df.loc[query]
        relations = list(df_selected['relation'])
        tails = list(df_selected['tail'])
        if keep_attr_react:
            relations_tails = [(relation,tails[index]) for index,relation in enumerate(relations) if (relation =='xReact' or relation == 'xAttr')]
            relations_tails = [(item[0],item[1]) for item in relations_tails if (item[1] == 'sad' or item[1] == 'upset')]
        if composed_p is not None:
            composed_rules = ddict(list)
            for text_pair in composed_p:
                group = 0
                for index,relation_tail in enumerate(relations_tails):
                    relation,tail = relation_tail
                    composed_rule = back_to_sen(text_pair,relation,tail)
                    composed_rules[group].append(composed_rule)
                    group += 1

            return composed_rules

        else:
            original_composed_rules = ddict(list)
            group = 0
            for index,relation_tail in enumerate(relations_tails):
                relation,tail = relation_tail
                composed_rule = back_to_sen(query,relation,tail)
                original_composed_rules[group].append(composed_rule)
                group += 1
            return original_composed_rules



    def select_composed_rules(self,query,composed_p,top_k_ratio = 0.73):
        composed_rules = self.query_relation_tail_ppl(query,composed_p)
        composed_rules_ppl_low = self.gptppl.have_ppl(composed_rules,top_k_ratio = top_k_ratio)

        return composed_rules_ppl_low


    def masked_composed_rules(self,original_composed_rules,composed_rules,top_k_jaccard):
        original_composed_rules_mask = ddict(list)
        composed_rules_mask = ddict(list)

        # TODO
        # Just mask the last token here which is not reasonable. May need to identify which part should be mask.
        mask_token = self.decoder.tokenizer.mask_token
        for key in original_composed_rules.keys():
            tmps = original_composed_rules[key]
            mask_sens = [tmp.rsplit(' ',1)[0] + ' ' + mask_token for tmp in tmps]
            original_composed_rules_mask[key] = mask_sens

        for key in composed_rules.keys():
            tmps = composed_rules[key]
            mask_sens = [tmp.rsplit(' ',1)[0] + ' ' + mask_token for tmp in tmps]
            composed_rules_mask[key] = mask_sens

        original_composed_rules_mask_softmaxs = self.decoder(original_composed_rules_mask)
        composed_rules_mask_softmaxs = self.decoder(composed_rules_mask)


        original_composed_rules_top_indices = self.decoder.top_k_for_jaccard(original_composed_rules_mask_softmaxs,top_k = top_k_jaccard)
        composed_rules_top_indices = self.decoder.top_k_for_jaccard(composed_rules_mask_softmaxs,top_k = top_k_jaccard)


        original_composed_rules_decoded_words = self.decoder.decode_to_word(original_composed_rules_top_indices)
        composed_rules_decoded_words = self.decoder.decode_to_word(composed_rules_top_indices)
        jaccard_result = self.decoder.jaccard(original_composed_rules_top_indices,composed_rules_top_indices)
        KL_result = self.decoder.KL_divergence(original_composed_rules_mask_softmaxs,composed_rules_mask_softmaxs)

        return jaccard_result,KL_result,original_composed_rules_decoded_words,composed_rules_decoded_words
