import sacrebleu

refs = [['The dog bit the bad man.']
       ]
sys = ['The dog bit the man.']

a = sacrebleu.corpus_bleu(sys, refs)
print(a)
print(a.score > 100)