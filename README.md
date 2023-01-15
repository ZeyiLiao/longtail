# longtail

**Ignore**
test
1

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


and replace the `build_index` function with
```
    def build_index(self, sentences_or_file_path: Union[str, List[str]],
                        use_faiss: bool = None,
                        faiss_fast: bool = False,
                        device: str = None,
                        batch_size: int = 64,
                        save_path: str = None):

        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True
            except:
                logger.warning("Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
                use_faiss = False

        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences

        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)

        logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}

        if use_faiss:
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences_or_file_path)))
            else:
                index = quantizer

            if (self.device == "cuda" and device != "cpu") or device == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(20 * 1024 * 1024 * 1024)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                else:
                    logger.info("Use CPU-version faiss")
            else:
                logger.info("Use CPU-version faiss")

            if faiss_fast:
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        np.save(save_path,index)
        logger.info("Finished")
```


Plz create a symbolic path to the following path and here are the path I used.

```
file_path = '/home/wangchunshu/preprocessed/all_heads'
save_embedding_path = 'zeyi/longtail_project/data/embeddings_all_heads.npy'
all_tuples_file = 'zeyi/longtail_project/data/all_tuples.csv'
```



Sample command at `sample_command.sh`


#TODO

1. Mask which word should be implemented at `masked_composed_rules()`
