from argparse import Namespace
from turtle import color
from helper import *
from retrieve import Retrieve
from negation_prompt import *
from write_plot import *



def generate_composed_p(query, text_pairs, combine_order, num_conjunction, negated = False):
    if negated:
        text_pairs = [text_pair for text_pair in text_pairs]
    if num_conjunction == 1:
        ans = []
        if combine_order == 'reverse':
            ans = [f'{text_pair} and {query}' for text_pair in text_pairs]
        elif combine_order == 'normal':
            ans = [f'{query} and {text_pair}' for text_pair in text_pairs]
        else:
            raise ("combine_order should be in ['normal','reverse']")
        return ans

    else:
        ans = []
        for sent in text_pairs:
            composed = [sent]
            for _ in range(num_conjunction - 1):
                rule = random.choice(text_pairs)
                while rule in composed:
                    rule = random.choice(text_pairs)
                composed.append(rule)
            if combine_order == 'reverse':
                all_rules = composed + [query]
                ans.append(' and '.join(all_rules))
            elif combine_order == 'normal':
                all_rules = [query] + composed
                ans.append(' and '.join(all_rules))
            else:
                raise ("combine_order should be in ['normal','reverse']")
        return ans


def main_NEP_data_process(args: Namespace, query, retrieve: Retrieve, negation_wrapper: PromptWrapper, NEP_rules_path: str):

    keep_attr_react = args.keep_attr_react


    original_composed_rules = retrieve.query_relation_tail_mask(query, keep_attr_react=keep_attr_react)
    negated_composed_rules = retrieve.query_relation_tail_mask(query, keep_attr_react=keep_attr_react,negation_wrapper = negation_wrapper)

    with open(f'./{NEP_rules_path}','a+') as f:
        for key in negated_composed_rules.keys():
            f.write(query)
            f.write('\t')
            f.write(original_composed_rules[key][0])
            f.write('\t')
            f.write(negated_composed_rules[key][0])
            f.write('\n')


            

def main_NEP(args,query,all_query,retrieve:Retrieve,query_order,output_file):
    top_k_jaccard = args.top_k_jaccard

    original_rule_dict = ddict(list)
    negated_rule_dict = ddict(list)

    for index,_ in enumerate(all_query[query]):
        original_rule_dict[index].append(_['original_rule'])
        negated_rule_dict[index].append(_['negated_rule'])

    original_results = retrieve.masked_composed_rules(original_rule_dict,top_k_jaccard)
    negated_results = retrieve.masked_composed_rules(negated_rule_dict,top_k_jaccard,split_from_mid = True)
    jaccard_result = retrieve.decoder.jaccard(original_results['top_indices'],negated_results['top_indices'])
    KL_result = retrieve.decoder.KL_divergence(original_results['softmaxs'],negated_results['softmaxs'])

    def return_None():
        return None

    final_output = ddict(return_None)
    final_output['jaccard'] = jaccard_result
    final_output['KL_div'] = KL_result
    final_output['original_results'] = original_results
    final_output['composed_results'] = negated_results
    final_output['original_rules'] = original_rule_dict
    final_output['composed_rules'] = negated_rule_dict

    file_info = dict()
    file_info['order'] = query_order
    file_info['output_file'] = output_file
    file_info['args'] = args
    file_info['query'] = query
    final_output['info'] = file_info

    jaccard, kl = plot_and_write(final_output,True)



def main_CPE(args: Namespace, query, retrieve: Retrieve, query_order, output_file):

    top_k_retrieval = args.top_k_retrieval
    threshold_retrieval = args.threshold_retrieval
    combine_order = args.combine_order
    top_k_composed_p = args.top_k_composed_p
    keep_attr_react = args.keep_attr_react
    top_k_jaccard = args.top_k_jaccard
    num_conjunction = args.num_conjunctions

    # 1.
    # retrieve all relevant sentence
    # filter out those whose label is not neutral in both direction
    neutral_texts = retrieve.query_neutral(query,
                                           top_k=top_k_retrieval,
                                           threshold=threshold_retrieval)

    # 2.
    # select neutral texts with high ppl when combining with query in a 'normal' order or 'reverse' order
    # named them composed p

    selected_neutral_texts = retrieve.select_highppl_neutral(
        query,
        neutral_texts,
        top_k=top_k_composed_p,
        combine_order=combine_order)
    composed_p = generate_composed_p(query, selected_neutral_texts,
                                     combine_order, num_conjunction)

    # 3.
    # select composed rules with low ppl
    # maybe useful in the future
    # composed_rules_ppl_low = retrieve.select_composed_rules(query,composed_p,top_k_ratio = 0.73)

    # 4.
    original_composed_rules = retrieve.query_relation_tail_mask(
        query, keep_attr_react=keep_attr_react)
    composed_rules = retrieve.query_relation_tail_mask(
        query, composed_p, keep_attr_react=keep_attr_react)

    original_rule_results = retrieve.masked_composed_rules(
        original_composed_rules,top_k_jaccard)

    composed_rule_results = retrieve.masked_composed_rules(
        composed_rules,top_k_jaccard)



    jaccard_result = retrieve.decoder.jaccard(original_rule_results['top_indices'],composed_rule_results['top_indices'])
    KL_result = retrieve.decoder.KL_divergence(original_rule_results['softmaxs'],composed_rule_results['softmaxs'])


    def return_None():
        return None

    final_output = ddict(return_None)

    final_output['jaccard'] = jaccard_result
    final_output['KL_div'] = KL_result
    final_output['original_results'] = original_rule_results
    final_output['composed_results'] = composed_rule_results
    final_output['original_rules'] = original_composed_rules
    final_output['composed_rules'] = composed_rules
    final_output['composed_p'] = composed_p

    file_info = dict()
    file_info['order'] = query_order
    file_info['output_file'] = output_file
    file_info['args'] = args
    file_info['query'] = query
    final_output['info'] = file_info

    jaccard, kl = plot_and_write(final_output)
    return jaccard, kl

def main(args: Namespace):
    all_heads_path = args.all_heads_path
    save_embedding_path = args.save_embedding_path
    all_tuples_path = args.all_tuples_path
    query_path = args.query_path
    save = args.save_embedding
    task = args.task
    if task == 'CPE':
        args.output_file = args.output_file + '/CPE'
    elif task == 'NEP':
        args.output_file = args.output_file + '/NEP'
    output_file = args.output_file



    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'We are using {device}')

    retrieve = Retrieve(all_heads_path, save_embedding_path, all_tuples_path,
                        device, task, save)

    if task == 'CPE':

        with open(query_path, 'r') as f:
            reader = csv.reader(f)
            jaccard_all = 0
            kl_all = 0
            num_queries = 0
            for index, query in enumerate(reader):
                query = query[0]
                print(f'Process for {query}')
                jaccard, kl = main_CPE(args, query, retrieve, index + 1,output_file)
                jaccard_all += jaccard
                kl_all += kl
                num_queries += 1


        file_path = f'./{output_file}/file_{args.combine_order}_{args.num_conjunctions}.txt'
        nl = '\n'
        with open(file_path, 'a+') as f:
            f.write(
                f'Final Averaged Jaccard Result for {args.num_conjunctions} conjunctions: {jaccard_all/num_queries} '
            )
            f.write(nl)
            f.write(
                f'Final Averaged KL Divergence for {args.num_conjunctions} conjunctions: {kl_all/num_queries} '
            )


    elif task == 'NEP':

        NEP_rules_path = args.NEP_rules_path
        top_k_jaccard = args.top_k_jaccard

        if not os.path.exists(NEP_rules_path):
            demonstration = "Given a statement, negate it to create a new sentence.\n"\
            "A: To see stars at night, it is better to turn on the lights.\n"\
            "B: The statement is false. To see stars at night, it is better not to turn on the lights.\n"\
            "A: Falling objects cannot accelerate beyond a certain speed.\n"\
            "B: The statement is false. Falling objects can accelerate beyond a certain speed.\n"\
            "A: People put a number before their names.\n"\
            "B: The statement is false. People do not put a number before their names."
            negation_wrapper = PromptWrapper(demonstration)
            with open(query_path,'r') as f:
                reader = csv.reader(f)
                for index, query in enumerate(reader):
                    query = query[0]
                    print(f'Process for {query}')
                    main_NEP_data_process(args, query, retrieve, negation_wrapper, NEP_rules_path)

        all_query = ddict(list)
        with open(f'{NEP_rules_path}') as f:
            reader = csv.reader(f,delimiter="\t")
            for line in reader:
                query = line[0]
                original_rule = line[1]
                negated_rule = line[2]
                tmp = dict()
                tmp['original_rule'] = original_rule
                tmp['negated_rule'] = negated_rule
                all_query[query].append(tmp)


        for index,query in enumerate(all_query.keys()):
            main_NEP(args,query,all_query,retrieve,index+1,output_file)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Longtailed project')
    parser.add_argument('--save_embedding',
                        action='store_true',
                        help='whether save the embedding')
    parser.add_argument('--all_heads_path', default='./data/all_heads')
    parser.add_argument('--save_embedding_path',
                        default='./data/embeddings_all_heads.npy')
    parser.add_argument('--all_tuples_path', default='./data/all_tuples.csv')
    parser.add_argument('--query_path',
                        help=" A file save the queries",
                        default='./input_file/query_CPE.csv')
    parser.add_argument('--task',
                        help='Do CPE or NPE probing',
                        choices=['NEP', 'CPE'])
    parser.add_argument('--NEP_rules_path',
                        help='the negation of the query_NEP.csv',
                        default='./input_file/query_NEP_rules.tsv')
    parser.add_argument('--output_file',
                        help='where to output',
                        default='./output_file')
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
                        default=5)
    parser.add_argument('--keep_attr_react',
                        help='Only focus on the xAttr and xReact relations',
                        action='store_false')

    parser.add_argument('--top_k_jaccard',
                        type=int,
                        help='Select top_k decoded words for jaccard',
                        default=5)
    parser.add_argument('--num_conjunctions',
                        type=int,
                        help='Select the number of conjunctions',
                        default=1)

    args = parser.parse_args()
    main(args)