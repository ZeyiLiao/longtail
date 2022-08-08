from helper import *
from nli import NLI
from gpt2_ppl import GPTppl
from roberta_decode import Decoder

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

def nli_data_format(query,keys):
    ans = []
    for key in keys:
        ans.append(query,key)
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

    def query_neutral_index(self,query,top_k=40,threshold=0.5):
        results = self.embedder.search(query,device = device, top_k=top_k, threshold=threshold)
        text = query
        text_pair = [text[0] for text in results]
        neutral_text_pair = self.nli(text,text_pair)
        return neutral_text_pair



    def top_neutral_sen(self,neutral_text,neutral_text_pair,top_k = 5,choice = 'normal'):
        sorted_text_pairs_index = self.gptppl.have_ppl_pair(neutral_text,neutral_text_pair,choice = choice,top_k = top_k)

        return sorted_text_pairs_index





    def top_composed_rules(self,neutral_text,sorted_text_pairs,top_k = 20):
        df_selected = self.df.loc[neutral_text]
        relations = list(df_selected['relation'])
        tails = list(df_selected['tail'])
        composed_rules = []
        for index,relation in enumerate(relations):

            tail = tails[index]
            for sort_text_pair in sorted_text_pairs:
                composed_rule = back_to_sen(sort_text_pair,relation,tail)
                composed_rules.append(composed_rule)


        sorted_text_index = self.gptppl.have_ppl(composed_rules,top_k = top_k)
        print('the ppl of the composed rules')
        for item in sorted_text_index.items():
            print(f'{composed_rules[item[0]]}   {item[1]}')

        return sorted_text_index,composed_rules

    def masked_composed_rules(self,query,sorted_text_pairs,masked_word):
        df_selected = self.df.loc[query]
        relations = list(df_selected['relation'])
        tails = list(df_selected['tail'])
        composed_rules = []
        for index,relation in enumerate(relations):

            tail = tails[index]
            for sort_text_pair in sorted_text_pairs:
                composed_rule = back_to_sen(sort_text_pair,relation,tail)
                composed_rules.append(composed_rule)

        decode_words = self.decoder.decode(composed_rules)
        assert(len(composed_rules) == len(decode_words))
        count = 0
        for i in range(len(decode_words)):
            if masked_word in decode_words[i]:
                count+= 1
            print(f'sens: {composed_rules[i]}    mask_word:{decode_words[i]}')
        precision = count / len(decode_words)

        print(f'Final match precision:{precision}')









file_path = '/home/wangchunshu/preprocessed/all_heads'
save_embedding_path = './data/embeddings_all_heads.npy'
all_tuples_file = './data/all_tuples.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'







retrieve = Retrieve(file_path,save_embedding_path,all_tuples_file,device)

# query = 'PersonX loses his father'
# mask_query = query.replace('loses','<mask>')
# masked_word = 'loses'

# query = 'PersonX decides to see a therapist'
# mask_query = query.replace('see','<mask>')
# masked_word = 'see'

query = 'PersonX never drives to Paris'
mask_query = query.replace('drives','<mask>')
masked_word = 'drives'


neutral_text_pair = retrieve.query_neutral_index(query,top_k = 40, threshold = 0.5)


choice = 'normal'
top_text_pairs_index = retrieve.top_neutral_sen(query,neutral_text_pair,top_k = 5,choice = choice)



# top_text_pairs_index 是用来取 neutral_text_pair里面的高ppl句子的

high_ppl_composed_p = []
if choice == 'reverse':
    for item in top_text_pairs_index.items():
        sen = f'{neutral_text_pair[item[0]]} and {query}'
        high_ppl_composed_p.append(f'{sen}')
        print(f'{sen}    {item[1]}'  )
elif choice == 'normal':
    for item in top_text_pairs_index.items():
        sen = f'{query} and {neutral_text_pair[item[0]]}'
        high_ppl_composed_p.append(f'{sen}')
        print(f'{sen}    {item[1]}'  )







# 尽管是reverse的，但我排列的时候还是按照query去取的relation and tail
# high_ppl_composed_rules_indexs 对于 all_composed_rules的高ppl的index
high_ppl_composed_rules_indexs,all_composed_rules = retrieve.top_composed_rules(query,high_ppl_composed_p,top_k = 20)
# high_ppl_composed_rules_indexs 对于 all_composed_rules的高ppl的index，所以可以用来取得 高ppl的 rules






high_ppl_composed_mask_p = []


if choice == 'reverse':
    for item in top_text_pairs_index.items():
        sen = f'{neutral_text_pair[item[0]]} and {mask_query}'
        high_ppl_composed_mask_p.append(f'{sen}')
        print(f'{sen}')
elif choice == 'normal':
    for item in top_text_pairs_index.items():
        sen = f'{mask_query} and {neutral_text_pair[item[0]]}'
        high_ppl_composed_mask_p.append(f'{sen}')
        print(f'{sen}')



retrieve.masked_composed_rules(query,high_ppl_composed_mask_p,masked_word)
