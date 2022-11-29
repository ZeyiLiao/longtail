#!/usr/bin/env bash

source activate /home/zeyi/miniconda3/envs/longtail

# PYTHONPATH=. python raw_generation_data/prepare_raw_data.py --input_file longtail_data/input.csv --save_file longtail_data/related_words.pkl \
# --csv_file longtail_data/ATOMIC10X_filter.csv --output_file longtail_data


extract.py
process.py
split_train_infer.py

# *****************************

python GPT_fill_task.py --inputs ../longtail_data/raw_data/property_centric/train_ids.csv \
--outputs ../longtail_data/for_finetune/property_centric --needed_count 3 --conti


# *****************************

python split.py --all_data ../property_centric_process/samples_process.jsonl --gpt_outputs_dir for_finetune/property_centric --data_type  w_m_t5
python split.py --all_data ../property_centric_process/samples_process.jsonl --gpt_outputs_dir for_finetune/property_centric --data_type  wo_m_t5
python split.py --all_data ../property_centric_process/samples_process.jsonl --gpt_outputs_dir for_finetune/property_centric --data_type  wo_m_gpt2

# *****************************
'''

source activate /home/zeyi/miniconda3/envs/tf_trainer

cd /home/zeyi/transformers

# property-centric 3b
rm -r ckpt/w_m_t5_3b_property_centric; PYTHONPATH=src USE_TF=0 deepspeed examples/pytorch/translation/run_translation.py --model_name_or_path t5-3b \
--output_dir ckpt/w_m_t5_3b_property_centric --overwrite_output_dir \
--source_lang input --target_lang output \
--train_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/w_m_t5/train/data.json --test_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/w_m_t5/test/data.json --validation_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/w_m_t5/dev/data.json \
--max_source_length 256 --max_target_length 256 --val_max_target_length 256 \
--do_train --num_train_epochs 4 --per_device_train_batch_size 8 --learning_rate 1e-4 \
--deepspeed tests/deepspeed/ds_config_zero3.json --save_strategy epoch --evaluation_strategy epoch \
--load_best_model_at_end --warmup_ratio 0.1 --adam_epsilon 1e-6 --weight_decay 0.01


rm -r ckpt/wo_m_t5_3b_property_centric; PYTHONPATH=src USE_TF=0 deepspeed examples/pytorch/translation/run_translation.py --model_name_or_path t5-3b \
--output_dir ckpt/wo_m_t5_3b_property_centric --overwrite_output_dir \
--source_lang input --target_lang output \
--train_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/wo_m_t5/train/data.json --test_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/wo_m_t5/test/data.json --validation_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/wo_m_t5/dev/data.json \
--max_source_length 256 --max_target_length 256 --val_max_target_length 256 \
--do_train --num_train_epochs 4 --per_device_train_batch_size 8 --learning_rate 1e-4 \
--deepspeed tests/deepspeed/ds_config_zero3.json --save_strategy epoch --evaluation_strategy epoch \
--load_best_model_at_end --warmup_ratio 0.1 --adam_epsilon 1e-6 --weight_decay 0.01



source activate pl_trainer

python main.py --data_dir /home/zeyi/longtail/longtail_data/for_finetune/property_centric --data_type wo_m_gpt2
'''
# **********************************************************


source activate /home/zeyi/miniconda3/envs/longtail
cd /home/zeyi/neurologic_decoding_lzy/seq2seq


# *****************************


# property centric
# disable early stopping, increse beta, increase alpha,   incerase the search space

# add a argument named change_format inorder to have different format


PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/translation/t5-3b-w_m_t5-319c8cee-7d5b-496c-b834-cd712eb16286 \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--min_tgt_length 5 --max_tgt_length 128 --bs 2 --beam_size 10 --length_penalty 0.1 --ngram_size 3 --prune_factor 500000 --beta 2 \
--early_stop 0 --save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/w_m_t5_3b.csv --n_obs 200 --parallelize


PYTHONPATH=.. python run_eval.py \
--model_name /home/zeyi/transformers_lzy/examples/pytorch/translation/t5-3b-w_m_t5-319c8cee-7d5b-496c-b834-cd712eb16286 \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--min_tgt_length 5 --max_tgt_length 128 \
--bs 8 --beam_size 10 --length_penalty 0.1 --ngram_size 3 \
--save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/w_m_t5_3b_vanilla.csv --n_obs 200 --parallelize


PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/translation/t5-3b-wo_m_t5-6eb26d90-6a7f-489c-8022-4b76432e50ed \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--min_tgt_length 5 --max_tgt_length 128 --bs 2 --beam_size 10 --length_penalty 0.1 --ngram_size 3 --prune_factor 500000 --beta 2 \
--early_stop 0 --save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_t5_3b.csv --n_obs 200 --parallelize --wo_mask


PYTHONPATH=.. python run_eval.py \
--model_name /home/zeyi/transformers_lzy/examples/pytorch/translation/t5-3b-wo_m_t5-6eb26d90-6a7f-489c-8022-4b76432e50ed  \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--min_tgt_length 5 --max_tgt_length 128 \
--bs 8 --beam_size 10 --length_penalty 0.1 --ngram_size 3 \
--save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_t5_3b_vanilla.csv --n_obs 200 --parallelize --wo_mask



# ********************************************************



python decode_pt.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/gpt2-large-wo_m_gpt2-74f21c1c-01f7-404a-974a-8bc730f5be77 \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gpt2_large.csv \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.1 \
--prune_factor 500000 --beta 2 --early_stop 10 --n_obs 200

python beam_search.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/gpt2-large-wo_m_gpt2-74f21c1c-01f7-404a-974a-8bc730f5be77 \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gpt2_large_vanilla.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.2 --n_obs 200


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
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.2 --n_obs 200





python decode_pt.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/gpt2-medium-wo_m_gpt2-f18c3d1f-fba5-4923-a8b8-6a3c2471cc07 \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gpt2_medium.csv \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.1 \
--prune_factor 500000 --beta 2 --early_stop 10 --n_obs 200

python beam_search.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/gpt2-medium-wo_m_gpt2-f18c3d1f-fba5-4923-a8b8-6a3c2471cc07 \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gpt2_medium_vanilla.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.2 --n_obs 200



python decode_pt.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/gpt2-wo_m_gpt2-19dfa65b-3228-4037-b317-83b2418c3600 \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gpt2.csv \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.1 \
--prune_factor 500000 --beta 2 --early_stop 10 --n_obs 200

python beam_search.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/gpt2-wo_m_gpt2-19dfa65b-3228-4037-b317-83b2418c3600 \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gpt2_vanilla.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.2 --n_obs 200


python decode_pt.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/gpt2-xl-wo_m_gpt2-6d0b8409-bc30-467f-89cf-ad53c6b6354a \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gpt2_xl.csv \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.1 \
--prune_factor 500000 --beta 2 --early_stop 10 --n_obs 200

python beam_search.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/gpt2-xl-wo_m_gpt2-6d0b8409-bc30-467f-89cf-ad53c6b6354a \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gpt2_xl_vanilla.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.2 --n_obs 200




python decode_pt.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/anton-l-gpt-j-tiny-random-wo_m_gpt2-61014616-9dcd-4cf7-a599-7148799a1c50 \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gptj-tiny.csv \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.1 \
--prune_factor 500000 --beta 2 --early_stop 10 --n_obs 200

python beam_search.py --model_name /home/zeyi/transformers_lzy/examples/pytorch/language-modeling/anton-l-gpt-j-tiny-random-wo_m_gpt2-61014616-9dcd-4cf7-a599-7148799a1c50 \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gptj-tiny_vanilla.csv \
--batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
--ngram_size 3 --length_penalty 0.2 --n_obs 200

# *****************************


# property centric
python GPT_fill_task.py --inputs ../longtail_data/raw_data/property_centric/infer_ids.csv \
--outputs ../longtail_data/generated_data/property_centric --needed_count 1 --conti --no_filter --n_obs 200


# *****************************




# *****************************
python easy_to_read.py --dir ./generated_data/property_centric


# *****************************