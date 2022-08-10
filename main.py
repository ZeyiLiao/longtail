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


def generate_composed_p(query,text_pairs,combine_order):
    ans = []
    if combine_order == 'reverse':
        ans = [f'{text_pair} and {query}' for text_pair in text_pairs]
    elif combine_order == 'normal':
        ans = [f'{query} and {text_pair}' for text_pair in text_pairs]
    else:
        raise ("combine_order should be in ['normal','reverse']")
    return ans


class Retrieve():
    def __init__(self,file_path,save_embedding_path,all_tuples_file,device):
        self.embedder = SimCSE('princeton-nlp/sup-simcse-bert-base-uncased',device = device)
        # embedder.build_index(file_path,device=device,batch_size=64,save_path=save_embedding_path)
        self.embedder.load_embeddings(file_path,save_embedding_path)
        self.nli = NLI(device)
        self.gptppl = GPTppl(device)
        self.decoder = Decoder(device)
        self.df = pd.read_csv(all_tuples_file,names = ['head','relation','tail'],index_col='head')

    def query_neutral(self,query,top_k=80,threshold=0.2):
        results = self.embedder.search(query,device = device, top_k=top_k, threshold=threshold)

        text_pair = [text[0] for text in results]
        neutral_text_pair = self.nli(query,text_pair)
        return neutral_text_pair



    def select_highppl_neutral(self,query,neutral_text_pair,top_k = 10,combine_order = 'normal'):
        selected_neutral_text_pair = self.gptppl.have_ppl_pair(query,neutral_text_pair,combine_order = combine_order,top_k = top_k)
        # composed_p = generate_composed_p(query,neutral_text_pair,selected_neutral_text_pair_index,combine_order)
        return selected_neutral_text_pair

    def  query_relation_tail_ppl(self,query,composed_p):
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

    def query_relation_tail_mask(self,query,composed_p = None ,keep_attr_react=False):
        df_selected = self.df.loc[query]
        relations = list(df_selected['relation'])
        if keep_attr_react:
            relations = [relation for relation in relations if (relation =='xReact' or relation == 'xAttr')]
        tails = list(df_selected['tail'])

        if composed_p is not None:
            composed_rules = ddict(list)
            for text_pair in composed_p:
                group = 0
                for index,relation in enumerate(relations):
                    tail = tails[index]
                    composed_rule = back_to_sen(text_pair,relation,tail)
                    composed_rules[group].append(composed_rule)
                    group += 1

            return composed_rules

        else:
            original_composed_rules = ddict(list)
            group = 0
            for index,relation in enumerate(relations):
                tail = tails[index]
                composed_rule = back_to_sen(query,relation,tail)
                original_composed_rules[group].append(composed_rule)
                group += 1
            return original_composed_rules



    def select_composed_rules(self,query,composed_p,top_k = 20):
        composed_rules = self.query_relation_tail_ppl(query,composed_p)
        composed_rules_ppl_low = self.gptppl.have_ppl(composed_rules,top_k = top_k)

        for item in composed_rules_ppl_low.items():
            print(f'{item[0]}   {item[1]}')

        return composed_rules_ppl_low


    def masked_composed_rules(self,original_composed_rules,composed_rules):
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


        original_composed_rules_top_indices = self.decoder.top_k_for_jaccard(original_composed_rules_mask_softmaxs,top_k = 5)
        composed_rules_top_indices = self.decoder.top_k_for_jaccard(composed_rules_mask_softmaxs,top_k = 5)

        jaccard_result = self.decoder.jaccard(original_composed_rules_top_indices,composed_rules_top_indices)
        KL_result = self.decoder.KL_divergence(original_composed_rules_mask_softmaxs,composed_rules_mask_softmaxs)

        return jaccard_result,KL_result

if __name__ == '__main__':

    file_path = '/home/wangchunshu/preprocessed/all_heads'
    save_embedding_path = './data/embeddings_all_heads.npy'
    all_tuples_file = './data/all_tuples.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'


    retrieve = Retrieve(file_path,save_embedding_path,all_tuples_file,device)

    query = 'PersonX loses his father'


# 1.
    # retrieve all relevant sentence
    # filter out those whose label is not neutral in both direction
    neutral_texts = retrieve.query_neutral(query,top_k = 200, threshold = 0.2)

# 2.
    # select neutral texts with high ppl when combining with query in a 'normal' order or 'reverse' order
    # named them composed p
    combine_order= 'normal'
    selected_neutral_texts = retrieve.select_highppl_neutral(query,neutral_texts,top_k = 10,combine_order = combine_order)
    composed_p = generate_composed_p(query,selected_neutral_texts,combine_order)

# 3.
    # select composed rules with low ppl
    composed_rules_ppl_low = retrieve.select_composed_rules(query,composed_p,top_k = 20)

    original_composed_rules = retrieve.query_relation_tail_mask(query,keep_attr_react = True)
    composed_rules = retrieve.query_relation_tail_mask(query,composed_p,keep_attr_react = True)
    jaccard_result,KL_result = retrieve.masked_composed_rules(original_composed_rules,composed_rules)
    print(jaccard_result)
    print(KL_result)
    nl = '\n'
    with open('./file.txt','w') as f:
        f.write('We also probe the GPT-J model with composed rules and the RHS score is perplexity, where the lower perplexity means the higher liklihood. ')
        f.write(nl)
        f.write('Note: When we generate composed p(instead of composed rules), we should rank the composed p by unliklihood, which means less plausible to model')
        f.write(nl)
        f.write(nl)
        f.write(nl)
        for item in composed_rules_ppl_low.items():
            f.write(item[0])
            f.write(f'         ppl:{item[1]}')
            f.write(nl)



        for key in original_composed_rules.keys():
            f.write(f'Rule{key}0(original):')
            f.write(original_composed_rules[key][0])
            f.write(nl)
            for index,sen in enumerate(composed_rules[key]):
                f.write(f'Rule{key}{index+1}(composed):')
                f.write(sen)
                f.write(nl)
            f.write(f'Jaccard score is {jaccard_result[key]}')
            f.write(nl)
            f.write(f'KL score is {KL_result[key]}')
            f.write(nl)
            f.write('*******************************')
            f.write(nl)
