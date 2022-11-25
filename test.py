import torch
from transformers import AutoModelForCausalLM,AutoConfig, AutoTokenizer

# a = torch.load('/home/zeyi/transformers/examples/pytorch/language-modeling/test-clm-stage2/pytorch_model-00001-of-00003.bin')
# b = torch.load('/home/zeyi/transformers/examples/pytorch/language-modeling/test-clm-stage2/pytorch_model-00002-of-00003.bin')
# m2 = AutoModelForCausalLM.from_pretrained('gpt2-large')
c = AutoConfig.from_pretrained('gpt2')
c.is_decoder = True
m = AutoModelForCausalLM.from_pretrained('gpt2',config = c)
t = AutoTokenizer.from_pretrained('gpt2')
text = 'I want to'
text_ids = t(text,return_tensors='pt')
m.generate(**text_ids)

print(m)
