from transformers import AutoTokenizer,AutoModelForCausalLM

t = AutoTokenizer.from_pretrained('t5-base')
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained('gpt2')
t.pad_token = t.eos_token
sent = ['I enjot walking with my dogs and','I want to sleep at']
ids = t(sent, return_tensors='pt',padding=True)
output = m.generate(**ids,min_length=5,num_beams = 3)
print(output)