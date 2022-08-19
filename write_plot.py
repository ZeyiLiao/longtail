from helper import *



def plot(original_composed_rules_masked_likelihood,
         composed_rules_masked_likelihood, original_composed_rules,output_file):


    for key in original_composed_rules.keys():

        original_likelihood = original_composed_rules_masked_likelihood[key]

        composed_likelihood = composed_rules_masked_likelihood[key]

        masked_word = original_composed_rules[key][0]

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(range(len(composed_likelihood)),
                composed_likelihood,
                label='Expanded rule',
                color='r')
        ax.plot(range(len(composed_likelihood)),
                original_likelihood *
                len(composed_likelihood),
                label='Original rule',
                linestyle='--',
                color='blue')
        ax.legend()
        ax.set_xlabel('Indice')
        ax.set_ylabel(f'Likelihood')
        # ax.set_ylim((0, 1))
        ax.set_title(f"Likelihood for '{masked_word}'")
        Path(f'./{output_file}/figure').mkdir(parents=True, exist_ok=True)
        fig.savefig(f'./{output_file}/figure/{masked_word}.png')



def plot_and_write(final_output,negated = False):
    file_info = final_output['info']
    args = file_info['args']


    # reformat the likelihood
    for key in final_output['original_results']['likelihood'].keys():
        final_output['original_results']['likelihood'][key] = torch.squeeze(final_output['original_results']['likelihood'][key],dim=-1).cpu().tolist()
        final_output['composed_results']['likelihood'][key] = torch.squeeze(final_output['composed_results']['likelihood'][key],dim=-1).cpu().tolist()


    if not negated:
        plot(final_output['original_results']['likelihood'],
            final_output['composed_results']['likelihood'], final_output['original_rules'],file_info['output_file'])


    nl = '\n'
    Path(f"./{file_info['output_file']}").mkdir(parents=True, exist_ok=True)
    file_path = f"./{file_info['output_file']}/file_{args.combine_order}_{args.num_conjunctions}.txt"
    with open(file_path, 'a+') as f:

        f.write(nl)
        f.write(nl)
        f.write(f"{file_info['order']}")
        f.write(nl)
        f.write("***Query:")
        f.write(nl)
        f.write(file_info['query'])
        f.write(nl)
        f.write(nl)
        f.write(nl)

        if final_output['composed_p'] is not None:
            f.write(
                '***Intermediate results whose are neutral in both direction and has high PPL***'
            )
            f.write(nl)
            f.write(nl)
            f.write(nl)
            for text in final_output['composed_p']:
                f.write(text)
                f.write(nl)

        f.write(nl)
        f.write('********************')
        f.write(nl)

        f.write(nl)
        if not negated:
            f.write(
                '***Mask the last word of composed rules and compute Jaccard , KL score compared to the original rules***'
            )
        else:
            f.write(
                '***Mask the last word of original rules and the same word in negated rules. Then compute Jaccard , KL score compared to the original rules***'
            )
        f.write(nl)
        f.write(nl)
        jaccard_composed = 0
        kl_div = 0
        num_composed = 0
        for key in final_output['original_rules'].keys():
            f.write(f'Rule{key} 0(original):')
            f.write(final_output['original_rules'][key][0])
            f.write(nl)
            f.write(f'Decoded words:  ')
            f.write(final_output['original_results']['decoded_words'][key][0])
            f.write(nl)
            f.write(f'Likelihood for masked token:  ')
            f.write(str(final_output['original_results']['likelihood'][key][0]))
            f.write(nl)
            f.write(nl)
            f.write('For box plot KL ')
            f.write(nl)
            for index, sen in enumerate(final_output['composed_rules'][key]):
                f.write(f"{final_output['KL_div'][key][index]},")
            f.write(nl)

            f.write('For box plot Jaccard')
            f.write(nl)
            for index, sen in enumerate(final_output['composed_rules'][key]):
                f.write(f"{final_output['jaccard'][key][index]},")
            f.write(nl)

            f.write(nl)
            f.write(nl)
            category = 'composed' if not negated else 'negated'
            # one query, multiple orignal rules. each original rule has multiple composed rule
            for index, sen in enumerate(final_output['composed_rules'][key]):
                f.write(f'Rule{key} {index+1}({category}):')
                f.write(sen)
                f.write(nl)
                f.write(f"Jaccard score: {final_output['jaccard'][key][index]}")
                jaccard_composed += final_output['jaccard'][key][index]
                kl_div += final_output['KL_div'][key][index]
                num_composed += 1
                f.write(nl)
                f.write(f"KL score: {final_output['KL_div'][key][index]}")
                f.write(nl)
                f.write(f'Decoded words:  ')
                f.write(final_output['composed_results']['decoded_words'][key][index])
                f.write(nl)
                f.write(f'Likelihood for masked token:  ')
                f.write(str(final_output['composed_results']['likelihood'][key][index]))
                f.write(nl)
                f.write(nl)
            f.write(nl)
            f.write('***********************************************')
            f.write(nl)
        f.write(nl)
        f.write(nl)
        f.write(
            f'Averaged Jaccard Result for {args.num_conjunctions} conjunctions: {jaccard_composed/num_composed} '
        )
        f.write(nl)
        f.write(
            f'Averaged KL Divergence for {args.num_conjunctions} conjunctions: {kl_div/num_composed} '
        )
        f.write(nl)
        return jaccard_composed / num_composed, kl_div / num_composed
