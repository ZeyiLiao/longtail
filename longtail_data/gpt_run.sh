
#!/usr/bin/env bash


python decode_pt.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/EleutherAI-gpt-j-6B-wo_m_gpt2-9c0f10b9-1744-43c5-8a96-11b1c30016fe \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gptj_6b.csv \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.1 \
--prune_factor 500000 --beta 2 --early_stop 10 --n_obs 200

python beam_search.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/EleutherAI-gpt-j-6B-wo_m_gpt2-9c0f10b9-1744-43c5-8a96-11b1c30016fe \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gptj_6b_vanilla.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.2 --n_obs 200


python decode_pt.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/EleutherAI-gpt-neox-20b-wo_m_gpt2-57d24e8a-804e-4faa-97e1-f193585d9dde \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gptneox_20b.csv \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--batch_size 2 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.1 \
--prune_factor 500000 --beta 2 --early_stop 10 --n_obs 200

python beam_search.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/EleutherAI-gpt-neox-20b-wo_m_gpt2-57d24e8a-804e-4faa-97e1-f193585d9dde \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gptneox_20b_vanilla.csv \
--batch_size 2 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.2 --n_obs 200