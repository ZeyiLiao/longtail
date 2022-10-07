from for_decode import *
import torch
import pickle
from collections import defaultdict as ddict
import matplotlib.pyplot as plt


def group(path_input,path_output,input_mask_token = '<extra_id_0>',desired_mask_token = '<mask>'):
    # 32的一半
    group_count = 16
    with open(path_input) as f:
        inputs = [x.rstrip() for x in f.readlines()]
    with open(path_output) as f:
        outputs = [x.rstrip() for x in f.readlines()]
    group_dict = ddict(lambda:ddict(list))
    group_dict_original = ddict(lambda:ddict(list))
    for i in range(0,len(inputs),group_count):
        mask_index = inputs[i].index(input_mask_token)
        key = inputs[i][:mask_index][:-5]
        tail = inputs[i][mask_index + len(input_mask_token):]
        be_masked_word = outputs[i].rsplit(' ',1)[1][:-1]
        wo_mask_outputs = outputs[i:i+group_count]
        w_mask_outputs = [_.rsplit(' ',1)[0] + ' ' + desired_mask_token + '.' for _ in wo_mask_outputs]
        group_dict[key][be_masked_word] = w_mask_outputs
        group_dict_original[key][be_masked_word].append(key + tail.rsplit(' ',1)[0] + ' ' + desired_mask_token + '.')
    return group_dict,group_dict_original





def masked_group_dict(group_dict,group_dict_original):

    group_dict_mask = ddict(list)
    group_dict_mask_word = ddict(list)


    group_dict_mask_softmaxs = decoder(group_dict)
    group_dict_original_mask_softmaxs = decoder(group_dict_original)

    # group_dict_masked_likelihood = decoder.mask_word_likelihood(group_dict_mask_softmaxs)
    # group_dict_original_masked_likelihood = decoder.mask_word_likelihood(group_dict_original_mask_softmaxs)

    kl_score = decoder.KL_divergence(group_dict_original_mask_softmaxs,group_dict_mask_softmaxs)



    return {'kl_score':kl_score}




def probing(path_input,path_output,task,desired_mask_token):
    group_dict,group_dict_original = group(path_input,path_output,desired_mask_token = desired_mask_token)
    group_dict,group_dict_original = dict(group_dict),dict(group_dict_original)
    with open(f'./group_dict_{task}.pkl','wb') as f:
        pickle.dump(group_dict,f)
    with open(f'./group_dict_original_{task}.pkl','wb') as f:
        pickle.dump(group_dict_original,f)

    with open(f'./group_dict_{task}.pkl','rb') as f:
        group_dict = pickle.load(f)
    with open(f'./group_dict_original_{task}.pkl','rb') as f:
        group_dict_original = pickle.load(f)


    results = masked_group_dict(group_dict,group_dict_original)
    kl_score = results['kl_score']
    kl_score = dict(kl_score)

    kl_score_all = []
    original_rule_all = []

    for key in kl_score.keys():
        for key2 in kl_score[key].keys():
            original_rule = group_dict_original[key][key2][0].replace(desired_mask_token,key2)
            original_rule_all.append(original_rule)
            kl_score_average = np.array(list(dict(kl_score[key][key2]).values())).mean()
            kl_score_all.append(kl_score_average)

    f = open(f'probe_results_{task}.txt','w')
    nl = '\n'
    f.write(f'Overall KL score is:')
    f.write(nl)
    f.write(str(np.array(kl_score_all).mean()))
    for index in range(len(kl_score_all)):
        f.write(nl)
        f.write(nl)
        f.write('Original rule:')
        f.write(nl)
        f.write(original_rule_all[index])
        f.write(nl)
        f.write(nl)
        f.write('average KL over corresponding new rules:')
        f.write(nl)
        f.write(str(kl_score_all[index]))
        f.write(nl)
        f.write(nl)
        f.write('************************')
        f.write(nl)
        f.write(nl)


    fig,ax = plt.subplots(figsize = (5,5))
    ax.plot(range(len(kl_score_all)),kl_score_all)
    ax.set_xlabel('# of original rules')
    ax.set_ylabel('Average KL score')
    fig.savefig(f'./KL_score_{task}.png')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# change it to large
decoder = Decoder(device,model_name = 'roberta-large')
desired_mask_token = decoder.tokenizer.mask_token

probing('/home/zeyi/longtail/Mturk_data/t5_input_w_mask.txt','/home/zeyi/longtail/Mturk_data/t5_3b_output_w_m.txt','t5_neuro',desired_mask_token)
probing('/home/zeyi/longtail/Mturk_data/t5_input_w_mask.txt','/home/zeyi/longtail/Mturk_data/t5_3b_output_vanilla_w_m.txt','t5_vanilla',desired_mask_token)
probing('/home/zeyi/longtail/Mturk_data/t5_input_w_mask.txt','/home/zeyi/longtail/Mturk_data/GPT_output_w_m.txt','gpt',desired_mask_token)