from helper import *
import pickle
import torchtext




embeddings_dict = {}
vector_all = []
with open("./counter_fit_embedding/counter-fitted-vectors.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
        vector_all.append(vector)


print(embeddings_dict['cat'])
with open("./counter_fit_embedding/counter_dict.pkl", "wb") as tf:
    pickle.dump(embeddings_dict,tf)




dim = 300
glove = torchtext.vocab.GloVe(name="840B",dim=dim)

dists_1 = torch.norm(glove.vectors - glove['flower'],dim=1)
dists_2 = torch.norm(glove.vectors - glove['sunlight'],dim=1)
dist = dists_1 + dists_2
dist = sorted(enumerate(dist) , key = lambda i: i[1])
for i in range(10):
    print(f'word : {glove.itos[dist[i][0]]},  similarity: {dist[i][1]}')
