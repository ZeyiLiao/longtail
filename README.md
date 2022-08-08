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



1. I first use `retrieve.query_neutral_index()` to select the similar and also neutral sentences for query.

2. Choose a order of ['normal','reverse'] to concat the query with the neutral senteces generated before.

3. `low_ppl_composed_p` stores the result of top_k composed_p (i.e. query + neutral sentence)

​      	4.1 Do the permuation between the `low_ppl_composed_p`  and (relation,tail)s of the query  and select top_k composed rules.

​		  4.2 Mask the query in `low_ppl_composed_p` and probe the model.
