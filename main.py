from argparse import Namespace
from turtle import color
from helper import *
from retrieve import Retrieve
from negation_prompt import *


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


def plot(original_composed_rules_masked_likelihood,
         composed_rules_masked_likelihood, original_composed_rules):
    original_composed_rules_masked_likelihood = torch.squeeze(
        original_composed_rules_masked_likelihood[0], dim=-1).cpu().tolist()
    composed_rules_masked_likelihood = torch.squeeze(
        composed_rules_masked_likelihood[0], dim=-1).cpu().tolist()
    masked_word = original_composed_rules[0]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(range(len(composed_rules_masked_likelihood)),
            composed_rules_masked_likelihood,
            label='Expanded rule',
            color='r')
    ax.plot(range(len(composed_rules_masked_likelihood)),
            original_composed_rules_masked_likelihood *
            len(composed_rules_masked_likelihood),
            label='Original rule',
            linestyle='--',
            color='blue')
    ax.legend()
    ax.set_xlabel('Indice')
    ax.set_ylabel(f'Likelihood')
    # ax.set_ylim((0, 1))
    ax.set_title(f'Likelihood for {masked_word}')
    fig.savefig(f'./figure/{masked_word}.png')

def main_NEP(args: Namespace, query, retrieve: Retrieve, negation_wrapper: PromptWrapper, query_order):
    top_k_retrieval = args.top_k_retrieval
    threshold_retrieval = args.threshold_retrieval
    combine_order = args.combine_order
    top_k_composed_p = args.top_k_composed_p
    keep_attr_react = args.keep_attr_react
    top_k_jaccard = args.top_k_jaccard
    num_conjunction = args.num_conjunctions

    original_composed_rules = retrieve.query_relation_tail_mask(query, keep_attr_react=keep_attr_react)
    negated_composed_rules = retrieve.query_relation_tail_mask(query, keep_attr_react=keep_attr_react,negation_wrapper = negation_wrapper)
    print(negated_composed_rules)

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


    plot(original_rule_results['likelihood'],
         composed_rule_results['likelihood'], original_composed_rules)

    # for json file, for future use

    # secondary_dict = ddict(list)
    # composed_p_dict = ddict(list)
    # for index,text in enumerate(composed_p):
    #     composed_p_dict[index] = text

    # # secondary_dict['composed p'] = composed_p_dict
    # secondary_dict['composed p'] = composed_p

    # original_rules_dict = dict()
    # tmp = dict()
    # for i,key in enumerate(original_composed_rules.keys()):
    #     kl_jaccard_rule_dict = dict()

    #     for index,text in enumerate(composed_rules[key]):
    #         kl_jaccard_rule_dict = {'composed_rule':text,'kl':KL_result[key][index],'jaccard':jaccard_result[key][index]}
    #         tmp[index] = kl_jaccard_rule_dict

    #     original_rules_dict[i] = {'original_rule':original_composed_rules[key][0],'result':tmp}

    # secondary_dict['original_rules'] = original_rules_dict
    # query_dict[query] = secondary_dict


    nl = '\n'
    file_path = f'.{output_file}/file_{combine_order}_{num_conjunction}.txt'
    with open(file_path, 'a+') as f:

        f.write(nl)
        f.write(nl)
        f.write(f'{query_order}')
        f.write(nl)
        f.write("***Query:")
        f.write(nl)
        f.write(query)
        f.write(nl)
        f.write(nl)
        f.write(nl)

        f.write(
            '***Intermediate results whose are neutral in both direction and has high PPL***'
        )
        f.write(nl)
        f.write(nl)
        f.write(nl)
        for text in composed_p:
            f.write(text)
            f.write(nl)

        f.write(nl)
        f.write('********************')
        f.write(nl)

        f.write(nl)
        f.write(
            '***Mask the last word of composed rules and compute Jaccard , KL score compared to the original rules***'
        )
        f.write(nl)
        jaccard_composed = 0
        kl_div = 0
        num_composed = 0
        for key in original_composed_rules.keys():
            f.write(f'Rule{key}0(original):')
            f.write(original_composed_rules[key][0])
            f.write(nl)
            f.write(f'Decoded words:  ')
            f.write(original_rule_results['decoded_words'][key][0])
            f.write(nl)
            f.write(nl)
            f.write('For box plot KL ')
            f.write(nl)
            for index, sen in enumerate(composed_rules[key]):
                f.write(f'{KL_result[key][index]},')
            f.write(nl)

            f.write('For box plot Jaccard')
            f.write(nl)
            for index, sen in enumerate(composed_rules[key]):
                f.write(f'{jaccard_result[key][index]},')
            f.write(nl)

            f.write(nl)
            f.write(nl)
            for index, sen in enumerate(composed_rules[key]):
                f.write(f'Rule{key}{index+1}(composed):')
                f.write(sen)
                f.write(nl)
                f.write(f'Jaccard score: {jaccard_result[key][index]}')
                jaccard_composed += jaccard_result[key][index]
                kl_div += KL_result[key][index]
                num_composed += 1
                f.write(nl)
                f.write(f'KL score: {KL_result[key][index]}')
                f.write(nl)
                f.write(f'Decoded words:  ')
                f.write(composed_rule_results['decoded_words'][key][index])
                f.write(nl)
                f.write(nl)
            f.write(nl)
            f.write('***********************************************')
            f.write(nl)
        f.write(nl)
        f.write(nl)
        f.write(
            f'Averaged Jaccard Result for {num_conjunction} conjunctions: {jaccard_composed/num_composed} '
        )
        f.write(nl)
        f.write(
            f'Averaged KL Divergence for {num_conjunction} conjunctions: {kl_div/num_composed} '
        )
        f.write(nl)
        return jaccard_composed / num_composed, kl_div / num_composed


def main(args: Namespace):
    all_heads_path = args.all_heads_path
    save_embedding_path = args.save_embedding_path
    all_tuples_path = args.all_tuples_path
    query_path = args.query_path
    save = args.save_embedding
    output_file = args.output_file

    task = args.task

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


        file_path = f'.{output_file}/file_{args.combine_order}_{args.num_conjunctions}.txt'
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
        NEP_pair_path = args.NEP_pair_path

        demonstration = "Given a statement, negate it to create a new sentence.\n"\
        "A: To see stars at night, it is better to turn on the lights.\n"\
        "B: The statement is false. To see stars at night, it is better not to turn on the lights.\n"\
        "A: Falling objects cannot accelerate beyond a certain speed.\n"\
        "B: The statement is false. Falling objects can accelerate beyond a certain speed.\n"\
        "A: People put a number before their names.\n"\
        "B: The statement is false. People do not put a number before their names."
        negation_wrapper = PromptWrapper(demonstration)

        if not os.path.exists(NEP_pair_path):
            with open(query_path,'r') as f:
                reader = csv.reader(f)
                for query in reader:
                    negation_process(query[0],NEP_pair_path,negation_wrapper)

        with open(query_path, 'r') as f:
            reader = csv.reader(f)
            for index, query in enumerate(reader):
                query = query[0]
                print(f'Process for {query}')
                main_NEP(args, query, retrieve, negation_wrapper, index + 1)



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
                        default='.input_file/query_CPE.csv')
    parser.add_argument('--task',
                        help='Do CPE or NPE probing',
                        choices=['NEP', 'CPE'])
    parser.add_argument('--NEP_pair_path',
                        help='the negation of the query_NEP.csv',
                        default='.input_file/query_NEP_pair.csv')
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