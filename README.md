# longtail

Plz add a code at Simcse/tool.py for not reloading the whole dataset with different querys.

```
    def load_embeddings(self,file_path,save_embedding_path):
        self.index= {"index":np.load(save_embedding_path)}
        sentences = []
        with open(file_path,"r") as f:
            logging.info("Loading sentences from %s ..." % (file_path))
            for line in tqdm(f):
                sentences.append(line.rstrip())
        self.index['sentences'] = sentences
```





Plz create a symbolic path to the follow path and here are the path I used.

```
file_path = '/home/wangchunshu/preprocessed/all_heads'
save_embedding_path = 'zeyi/longtail_project/data/embeddings_all_heads.npy'
all_tuples_file = 'zeyi/longtail_project/data/all_tuples.csv'
```
