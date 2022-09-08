from helper import *
from argparse import Namespace
from retrieve import Retrieve
from neutral_dis import filter_neutral



def dependency_parse(predictor,retrieve,selected_neutral_texts,query,output_file):
    prompt_file = f'{output_file}/pt.txt'
    constraint_file = f'{output_file}/constraint.json'

    for neutral_text in selected_neutral_texts:
        dp = predictor.predict(
            sentence=neutral_text
        )
        words = dp['words']
        noun_indices = dp['pos']

        Nouns = []

        for index,_ in enumerate(noun_indices):
            if _ == 'NOUN':
                Nouns.append(words[index])
        similarity_scores = retrieve.embedder.similarity(query,Nouns)
        constraints_words = [Nouns[i] for i,score in enumerate(similarity_scores) if score > 0.3]

        if len(constraints_words) == 0:
            continue

        desired_constraint = []
        for word in constraints_words:
            tmp = []
            tmp.append(word)
            desired_constraint.append(tmp)

        with open(constraint_file,'a+') as f:
            json.dump(desired_constraint,f)
            f.write('\n')

        with open(prompt_file,'a+') as f:
            f.write(query + ' and')
            f.write('\n')




def main(args: Namespace):
    all_heads_path = args.all_heads_path
    save_embedding_path = args.save_embedding_path
    all_tuples_path = args.all_tuples_path
    query_path = args.query_path
    save = args.save_embedding
    task = args.task
    if task == 'CPE':
        args.output_file = args.output_file + '/constraints'

    output_file = args.output_file
    Path(output_file).mkdir(parents=True, exist_ok=True)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'We are using {device}')

    retrieve = Retrieve(all_heads_path, save_embedding_path, all_tuples_path,
                        device, task, save)

    # predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")


    if task == 'CPE':
        top_k_retrieval = args.top_k_retrieval
        threshold_retrieval = args.threshold_retrieval
        combine_order = args.combine_order
        top_k_composed_p = args.top_k_composed_p

        # with open(query_path, 'r') as f:
        #     reader = csv.reader(f)
        #     for index, query in enumerate(reader):
        #         query = query[0]
        #         print(f'Process for: {query}')
        #         neutral_texts = retrieve.query_neutral(query,
        #                                 top_k=top_k_retrieval,
        #                                 threshold=threshold_retrieval)

        #         selected_neutral_texts = retrieve.select_highppl_neutral(query,
        #                                                             neutral_texts,
        #                                                             top_k=top_k_composed_p,
        #                                                             combine_order=combine_order)

        #         dependency_parse(predictor,retrieve,selected_neutral_texts,query,output_file)



        pair_dict = filter_neutral(retrieve)
        nl = '\n'
        with open('../output_file/constraints/decoded_result_filter_by_neutralness.txt','a+') as f:
            for i,query in enumerate(pair_dict.keys()):
                f.write(f'Query{i}:  {query}')
                f.write(nl)
                f.write(nl)
                f.write('Insertation filtered by neutralness')
                f.write(nl)
                if len(pair_dict[query]['decoded']) == 0:
                    f.write('After filtering, there are no more satisfied neutral insertation. ')
                    f.write(nl)
                    f.write(nl)
                else:
                    for index,sen in enumerate(pair_dict[query]['decoded']):
                        f.write(f'Insertation {index}:')
                        f.write(nl)
                        f.write(nl)
                        f.write(f"constraints:{pair_dict[query]['constraints'][index]}")
                        f.write(nl)
                        f.write(nl)
                        f.write(sen)
                        f.write(nl)
                        f.write(nl)
                        f.write(nl)

                f.write('******************************************************')
                f.write(nl)
                f.write(nl)
                f.write(nl)




        # for mask token predition
        # for query in pair_dict.keys():
        #     if len(pair_dict[query]) == 0:
        #         continue
        #     original_composed_rules = retrieve.query_relation_tail_mask(query, keep_attr_react=True)
        #     composed_rules = retrieve.query_relation_tail_mask(query, pair_dict[query], keep_attr_react=True)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Longtailed project')
    parser.add_argument('--save_embedding',
                        action='store_true',
                        help='whether save the embedding')
    parser.add_argument('--all_heads_path', default='../data/all_heads')
    parser.add_argument('--save_embedding_path',
                        default='./data/embeddings_all_heads.npy')
    parser.add_argument('--all_tuples_path', default='../data/all_tuples.csv')
    parser.add_argument('--query_path',
                        help=" A file save the queries",
                        default='../input_file/query_CPE.csv')
    parser.add_argument('--task',
                        help='Do CPE or NPE probing',
                        choices=['NEP', 'CPE'])
    parser.add_argument('--output_file',
                        help='where to output',
                        default='../output_file')
    parser.add_argument('--top_k_retrieval',
                        type=int,
                        help='Select top_k during retrieval',
                        default=400)
    parser.add_argument('--threshold_retrieval',
                        type=float,
                        help='Threshold for retrieval',
                        default=0.2)
    parser.add_argument('--combine_order',
                        help="Combine the p and p' in two order",
                        choices=['normal', 'reverse'])
    parser.add_argument(
                        '--top_k_composed_p',
                        type=int,
                        help=
                        'Select top_k composed_p from top to end which are not that plausible',
                        default=3)
    args = parser.parse_args()
    main(args)
