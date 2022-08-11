from helper import *
from retrieve import Retrieve



def generate_composed_p(query,text_pairs,combine_order):
    ans = []
    if combine_order == 'reverse':
        ans = [f'{text_pair} and {query}' for text_pair in text_pairs]
    elif combine_order == 'normal':
        ans = [f'{query} and {text_pair}' for text_pair in text_pairs]
    else:
        raise ("combine_order should be in ['normal','reverse']")
    return ans


def main_process(args,query,retrieve):

    top_k_retrieval = args.top_k_retrieval
    threshold_retrieval = args.threshold_retrieval
    combine_order= args.combine_order
    top_k_composed_p = args.top_k_composed_p
    keep_attr_react = args.keep_attr_react
    top_k_jaccard = args.top_k_jaccard


# 1.
    # retrieve all relevant sentence
    # filter out those whose label is not neutral in both direction
    neutral_texts = retrieve.query_neutral(query,top_k = top_k_retrieval, threshold = threshold_retrieval)

# 2.
    # select neutral texts with high ppl when combining with query in a 'normal' order or 'reverse' order
    # named them composed p

    selected_neutral_texts = retrieve.select_highppl_neutral(query,neutral_texts,top_k = top_k_composed_p,combine_order = combine_order)
    composed_p = generate_composed_p(query,selected_neutral_texts,combine_order)

# 3.
    # select composed rules with low ppl
    # maybe useful in the future
    # composed_rules_ppl_low = retrieve.select_composed_rules(query,composed_p,top_k_ratio = 0.73)

# 4.
    original_composed_rules = retrieve.query_relation_tail_mask(query,keep_attr_react = keep_attr_react)
    composed_rules = retrieve.query_relation_tail_mask(query,composed_p,keep_attr_react = keep_attr_react)

    jaccard_result,KL_result,original_composed_rules_decoded_words,composed_rules_decoded_words = retrieve.masked_composed_rules(original_composed_rules,composed_rules,top_k_jaccard)



    nl = '\n'
    file_path = f'./file_{combine_order}.txt'
    with open(file_path,'a+') as f:

        f.write("Query:")
        f.write(nl)
        f.write(query)
        f.write(nl)
        f.write(nl)
        f.write(nl)


        f.write('Intermediate results whose are neutral in both direction and has high PPL')
        f.write(nl)
        f.write(nl)
        f.write(nl)
        for text in composed_p:
            f.write(text)
            f.write(nl)


        f.write('********************')


        f.write(nl)
        f.write('Mask the last word of composed rules and compute Jaccard , KL score compared to the original rules')
        f.write(nl)
        for key in original_composed_rules.keys():
            f.write(f'Rule{key}0(original):')
            f.write(original_composed_rules[key][0])
            f.write(nl)
            f.write(f'Decoded words:  ')
            f.write(original_composed_rules_decoded_words[key][0])
            f.write(nl)
            f.write(nl)
            for index,sen in enumerate(composed_rules[key]):
                f.write(f'Rule{key}{index+1}(composed):')
                f.write(sen)
                f.write(nl)
                f.write(f'Jaccard score: {jaccard_result[key][index]}')
                f.write(nl)
                f.write(f'KL score: {KL_result[key][index]}')
                f.write(nl)
                f.write(f'Decoded words:  ')
                f.write(composed_rules_decoded_words[key][index])
                f.write(nl)
                f.write(nl)
            f.write(nl)
            f.write('***********************************************')
            f.write(nl)
        f.write(nl)
        f.write(nl)
        f.write('*************************************************************************************')


def main(args):
    all_heads_path = args.all_heads_path
    save_embedding_path = args.save_embedding_path
    all_tuples_path = args.all_tuples_path
    query_path = args.query_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    retrieve = Retrieve(all_heads_path,save_embedding_path,all_tuples_path,device)
    with open(query_path,'r') as f:
        reader = csv.reader(f)
        for query in tqdm(reader):
            query = query[0]
            print(f'Process for {query}')
            main_process(args,query,retrieve)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Longtailed project')
    parser.add_argument('--all_heads_path', default= './data/all_heads')
    parser.add_argument('--save_embedding_path', default= './data/embeddings_all_heads.npy')
    parser.add_argument('--all_tuples_path', default='./data/all_tuples.csv')
    parser.add_argument('--query_path',help=" A file save the queries", default= './query.csv')
    parser.add_argument('--top_k_retrieval', help= 'Select top_k during retrieval', default = 400)
    parser.add_argument('--threshold_retrieval', help = 'Threshold for retrieval', default= 0.2)
    parser.add_argument('--combine_order', help = "Combine the p and p' in two order", choices= ['normal','reverse'])
    parser.add_argument('--top_k_composed_p', help = 'Select top_k composed_p from top to end which are not that plausible',default=10)
    parser.add_argument('--keep_attr_react', help = 'Only focus on the xAttr and xReact relations', action = 'store_false')
    parser.add_argument('--top_k_jaccard', help= 'Select top_k decoded words for jaccard', default = 3)

    args = parser.parse_args()
    main(args)