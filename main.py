from turtle import color
from helper import *
from retrieve import Retrieve




def generate_composed_p(query,text_pairs,combine_order,num_conjunction):
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
            for _ in range (num_conjunction-1):
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


def plot(original_composed_rules_masked_likelihood,composed_rules_masked_likelihood,composed_rules_masked_word):
    original_composed_rules_masked_likelihood = torch.squeeze(original_composed_rules_masked_likelihood[0],dim=-1).cpu().tolist()
    composed_rules_masked_likelihood = torch.squeeze(composed_rules_masked_likelihood[0],dim=-1).cpu().tolist()
    masked_word = composed_rules_masked_word[0]

    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(range(len(composed_rules_masked_likelihood)),composed_rules_masked_likelihood,label='Expanded rule',color = 'r')
    ax.plot(range(len(composed_rules_masked_likelihood)),original_composed_rules_masked_likelihood * len(composed_rules_masked_likelihood),label = 'Original rule',linestyle = '--', color = 'blue')
    ax.legend()
    ax.set_xlabel('Indice')
    ax.set_ylabel(f'Likelihood')
    ax.set_ylim((0,1))
    ax.set_title(f'Likelihood for {masked_word}')
    fig.savefig(f'./figure/{masked_word}.png')


def plot(original_composed_rules_masked_likelihood,composed_rules_masked_likelihood,composed_rules_masked_word):
    original_composed_rules_masked_likelihood = torch.squeeze(original_composed_rules_masked_likelihood[0],dim=-1).cpu().tolist()
    composed_rules_masked_likelihood = torch.squeeze(composed_rules_masked_likelihood[0],dim=-1).cpu().tolist()
    masked_word = composed_rules_masked_word[0]

    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(range(len(composed_rules_masked_likelihood)),composed_rules_masked_likelihood,label='Expanded rule',color = 'r')
    ax.plot(range(len(composed_rules_masked_likelihood)),original_composed_rules_masked_likelihood * len(composed_rules_masked_likelihood),label = 'Original rule',linestyle = '--', color = 'blue')
    ax.legend()
    ax.set_xlabel('Indice')
    ax.set_ylabel(f'Likelihood')
    ax.set_ylim((0,1))
    ax.set_title(f'Likelihood for {masked_word}')
    fig.savefig(f'./figure/{masked_word}.png')


def main_process(args,query,retrieve: Retrieve,query_order):

    top_k_retrieval = args.top_k_retrieval
    threshold_retrieval = args.threshold_retrieval
    combine_order= args.combine_order
    top_k_composed_p = args.top_k_composed_p
    keep_attr_react = args.keep_attr_react
    top_k_jaccard = args.top_k_jaccard
    num_conjunction = args.num_conjunctions

# 1.
    # retrieve all relevant sentence
    # filter out those whose label is not neutral in both direction
    neutral_texts = retrieve.query_neutral(query,top_k = top_k_retrieval, threshold = threshold_retrieval)

# 2.
    # select neutral texts with high ppl when combining with query in a 'normal' order or 'reverse' order
    # named them composed p

    selected_neutral_texts = retrieve.select_highppl_neutral(query,neutral_texts,top_k = top_k_composed_p,combine_order = combine_order)
    composed_p = generate_composed_p(query,selected_neutral_texts,combine_order, num_conjunction)

# 3.
    # select composed rules with low ppl
    # maybe useful in the future
    # composed_rules_ppl_low = retrieve.select_composed_rules(query,composed_p,top_k_ratio = 0.73)

# 4.
    original_composed_rules = retrieve.query_relation_tail_mask(query,keep_attr_react = keep_attr_react)
    composed_rules = retrieve.query_relation_tail_mask(query,composed_p,keep_attr_react = keep_attr_react)

    result,decoded_words,likelihood,composed_rules_masked_word = retrieve.masked_composed_rules(original_composed_rules,composed_rules,top_k_jaccard)

    jaccard_result,KL_result = result[0],result[1]
    original_composed_rules_decoded_words,composed_rules_decoded_words = decoded_words[0],decoded_words[1]
    original_composed_rules_masked_likelihood,composed_rules_masked_likelihood = likelihood[0],likelihood[1]



    plot(original_composed_rules_masked_likelihood,composed_rules_masked_likelihood,composed_rules_masked_word)
    nl = '\n'
    file_path = f'./file_{combine_order}_{num_conjunction}.txt'
    with open(file_path,'a+') as f:

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


        f.write('***Intermediate results whose are neutral in both direction and has high PPL***')
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
        f.write('***Mask the last word of composed rules and compute Jaccard , KL score compared to the original rules***')
        f.write(nl)
        jaccard_composed = 0
        kl_div = 0
        num_composed = 0
        for key in original_composed_rules.keys():
            f.write(f'Rule{key}0(original):')
            f.write(original_composed_rules[key][0])
            f.write(nl)
            f.write(f'Decoded words:  ')
            f.write(original_composed_rules_decoded_words[key][0])
            f.write(nl)
            f.write(nl)
            f.write('For box plot KL ')
            f.write(nl)
            for index,sen in enumerate(composed_rules[key]):
                f.write(f'{KL_result[key][index]},')
            f.write(nl)

            f.write('For box plot Jaccard')
            f.write(nl)
            for index,sen in enumerate(composed_rules[key]):
                f.write(f'{jaccard_result[key][index]},')
            f.write(nl)

            f.write(nl)
            f.write(nl)
            for index,sen in enumerate(composed_rules[key]):
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
                f.write(composed_rules_decoded_words[key][index])
                f.write(nl)
                f.write(nl)
            f.write(nl)
            f.write('***********************************************')
            f.write(nl)
        f.write(nl)
        f.write(nl)
        f.write(f'Averaged Jaccard Result for {num_conjunction} conjunctions: {jaccard_composed/num_composed} ')
        f.write(nl)
        f.write(f'Averaged KL Divergence for {num_conjunction} conjunctions: {kl_div/num_composed} ')
        f.write(nl)
        return jaccard_composed/num_composed, kl_div/num_composed



def main(args):
    all_heads_path = args.all_heads_path
    save_embedding_path = args.save_embedding_path
    all_tuples_path = args.all_tuples_path
    query_path = args.query_path
    save = args.save_embedding

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'We are using {device}')



    retrieve = Retrieve(all_heads_path,save_embedding_path,all_tuples_path,device,save)
    with open(query_path,'r') as f:
        reader = csv.reader(f)
        jaccard_all = 0
        kl_all = 0
        num_queries = 0
        for index,query in enumerate(reader):
            query = query[0]
            print(f'Process for {query}')
            jaccard, kl =  main_process(args,query,retrieve,index+1)
            jaccard_all += jaccard
            kl_all += kl
            num_queries += 1

    file_path = f'./file_{args.combine_order}_{args.num_conjunctions}.txt'
    with open(file_path,'a+') as f:
        f.write(f'Final Averaged Jaccard Result for {args.num_conjunctions} conjunctions: {jaccard_all/num_queries} ')
        f.write(nl)
        f.write(f'Final Averaged KL Divergence for {args.num_conjunctions} conjunctions: {kl_all/num_queries} ')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Longtailed project')
    parser.add_argument('--save_embedding',action = 'store_true',help = 'whether save the embedding')
    parser.add_argument('--all_heads_path', default= './data/all_heads')
    parser.add_argument('--save_embedding_path', default= './data/embeddings_all_heads.npy')
    parser.add_argument('--all_tuples_path', default='./data/all_tuples.csv')
    parser.add_argument('--query_path',help=" A file save the queries", default= './query.csv')
    parser.add_argument('--top_k_retrieval',type =int, help= 'Select top_k during retrieval', default = 400)
    parser.add_argument('--threshold_retrieval', type =float, help = 'Threshold for retrieval', default= 0.2)
    parser.add_argument('--combine_order', help = "Combine the p and p' in two order", choices= ['normal','reverse'])
    parser.add_argument('--top_k_composed_p',  type =int, help = 'Select top_k composed_p from top to end which are not that plausible',default=40)
    parser.add_argument('--keep_attr_react', help = 'Only focus on the xAttr and xReact relations', action = 'store_false')

    parser.add_argument('--top_k_jaccard', type =int, help= 'Select top_k decoded words for jaccard', default = 10)
    parser.add_argument('--num_conjunctions', type =int, help= 'Select the number of conjunctions', default = 1)

    args = parser.parse_args()
    main(args)