from transformers import AutoTokenizer,AutoModelForCausalLM

name = "gpt2"
t = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name)
''
sent = "I want to go"
enc = t(sent,return_tensors= "pt")
generation = m.generate(**enc,typical_p = 0.9,do_sample = True)