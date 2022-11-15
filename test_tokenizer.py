from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,AutoModelForCausalLM,AutoModel


m_t_map = {}


t_t5 = AutoTokenizer.from_pretrained('t5-base')
t_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
t_gpt = AutoTokenizer.from_pretrained('gpt2')
t_bart = AutoTokenizer.from_pretrained('facebook/bart-base')


m_t5 = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
m_bart = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')
m_gpt = AutoModelForCausalLM.from_pretrained('gpt2')
m_bert = AutoModel.from_pretrained('bert-base-uncased')

tmp_d = {'model':m_t5,'tokenizer':t_t5}
m_t_map['t5'] = tmp_d

tmp_d = {'model':m_bert,'tokenizer':t_bert}
m_t_map['bert'] = tmp_d

tmp_d = {'model':m_bart,'tokenizer':t_bart}
m_t_map['bart'] = tmp_d

tmp_d = {'model':m_gpt,'tokenizer':t_gpt}
m_t_map['gpt2'] = tmp_d


l = ["t5","gpt2","bart"]
sen = 'I want to sleep and </s>'


for _t in l:
    print(str(_t))
    t = m_t_map[_t]['tokenizer']
    print(t.tokenize(sen))
    
    print(t.tokenize('want'))
    print(t(sen,return_tensors = 'pt'))
    print(t([sen],return_tensors = 'pt'))

    ids = t(sen).input_ids
    print(ids)
    print(t.convert_ids_to_tokens(ids))
    print(t.decode(ids,skip_special_tokens = True,clean_up_tokenization_spaces= False))



    print('See top-p decoding')
    print('*'* 100)
    m = m_t_map[_t]['model']
    output = m.generate(
    **t(sen,return_tensors = 'pt'),
    do_sample=True, 
    max_length=50, 
    top_p=0.92, 
    top_k=0
    )

    print(t.convert_ids_to_tokens(output[0]))
    print(t.batch_decode(output))
    print(t.batch_decode(output,skip_special_tokens = True,clean_up_tokenization_spaces= True))
    print()
    print()
    print()

    



'''
# https://huggingface.co/docs/transformers/tokenizer_summary#bytepair-encoding-bpe

gpt BPE
t5 sentence piece
bert wordpiece

so when we deal with gpt decoding, we need to add 'G' before the token while dont need for t5.
https://github.com/GXimingLu/neurologic_decoding/blob/9d33b871c9633ca360bec1ce45dc4a0bc24b537a/zero_shot/decode_pt.py#L70


'''