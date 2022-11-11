import jsonlines
import sacrebleu

def move_conjunction_forward(path):
    l = []
    with jsonlines.open(path) as f:
        for line in f:
            l.append(line)
    return l

					

						



o_path_jsonl = '/home/zeyi/longtail/longtail_data/generated_data/property_centric/compare.jsonl'

l = move_conjunction_forward(o_path_jsonl)
l = sorted(l, key = lambda _ : _['Constraints_length'],reverse=True)

nl = '\n'

with open('/home/zeyi/longtail/longtail_data/generated_data/property_centric/compare_long_first.txt','w') as fo:
    for instance in l:
        base = instance['Base']
        id = instance['id']
        cons = instance['Constraints']
        sample_conti = instance['Sample_continuation']
        fo.write(f'Base: {base} ,  id :{id}' )
        fo.write(nl)
        fo.write(f'Constraints: {cons}')
        fo.write(nl)
        fo.write(f'Sample continuation: {sample_conti}')
        fo.write(nl)
        fo.write(nl)
        neuro = instance['neruo']
        vanilla = instance['vanilla']
        GPT_3 = instance['GPT3']
        fo.write(f'Neuro: {neuro}')
        fo.write(nl)
        fo.write(f'Vanilla: {vanilla}')
        fo.write(nl)
        bleu_score = sacrebleu.corpus_bleu([neuro], [[vanilla]]).score
        fo.write(f'bleu: {bleu_score}')
        fo.write(nl)

        fo.write(f'GPT3: {GPT_3}')
        fo.write(nl)

        fo.write(nl)
        fo.write(nl)
        fo.write('*******************************')
        fo.write(nl)
        fo.write(nl)
        fo.write(nl)