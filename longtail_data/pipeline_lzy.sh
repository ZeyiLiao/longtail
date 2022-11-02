#!/bin/bash

source activate /home/zeyi/miniconda3/envs/longtail

PYTHONPATH=. python raw_generation_data/prepare_raw_data.py --input_file longtail_data/input.csv --save_file longtail_data/related_words.pkl \
--csv_file longtail_data/ATOMIC10X_filter.csv --output_file longtail_data


# *****************************
PYTHONPATH=. python GPT_task/GPT_fill_task.py --inputs longtail_data/raw_data/for_dis/inputs_t5_train.csv \
--lemma_constraints longtail_data/raw_data/for_dis/lemma_constraints_t5_train.json \
--inflection_constraints longtail_data/raw_data/for_dis/inflection_constraints_t5_train.json \
--outputs longtail_data/for_finetune/for_dis --needed_count 3

# for continuation format
PYTHONPATH=. python GPT_task/GPT_fill_task.py --inputs longtail_data/raw_data/conti/inputs_t5_train.csv \
--lemma_constraints longtail_data/raw_data/conti/lemma_constraints_t5_train.json \
--inflection_constraints longtail_data/raw_data/conti/inflection_constraints_t5_train.json \
--outputs longtail_data/for_finetune/conti --needed_count 3 --conti



PYTHONPATH=. python GPT_task/GPT_fill_task.py --inputs longtail_data/raw_data/property_centric/inputs_t5_train.csv \
--lemma_constraints longtail_data/raw_data/property_centric/lemma_constraints_t5_train.json \
--inflection_constraints longtail_data/raw_data/property_centric/inflection_constraints_t5_train.json \
--outputs longtail_data/for_finetune/property_centric --needed_count 3 --conti



# *****************************












# *****************************
PYTHONPATH=. python longtail_data/splitting.py --gpt_outputs_dir longtail_data/for_finetune/for_dis

PYTHONPATH=. python longtail_data/splitting.py --gpt_outputs_dir longtail_data/for_finetune/conti


PYTHONPATH=. python longtail_data/splitting.py --gpt_outputs_dir longtail_data/for_finetune/property_centric

# *****************************







source activate /home/zeyi/miniconda3/envs/tf_trainer

cd /home/zeyi/transformers

rm -r ckpt/t5_3b_w_m_dis; PYTHONPATH=src USE_TF=0 deepspeed examples/pytorch/translation/run_translation.py --model_name_or_path t5-3b \
--output_dir ckpt/t5_3b_w_m_dis --overwrite_output_dir \
--source_lang input --target_lang output \
--train_file /home/zeyi/longtail/longtail_data/for_finetune/for_dis/w_m_t5/train/data.json --test_file /home/zeyi/longtail/longtail_data/for_finetune/for_dis/w_m_t5/test/data.json --validation_file /home/zeyi/longtail/longtail_data/for_finetune/for_dis/w_m_t5/dev/data.json \
--max_source_length 256 --max_target_length 256 --val_max_target_length 256 \
--do_train --num_train_epochs 4 --per_device_train_batch_size 8 --learning_rate 1e-4 \
--deepspeed tests/deepspeed/ds_config_zero3.json --save_strategy epoch --evaluation_strategy epoch \
--load_best_model_at_end --warmup_ratio 0.1 --adam_epsilon 1e-6 --weight_decay 0.01

# for conti
rm -r ckpt/t5_3b_w_m_conti; PYTHONPATH=src USE_TF=0 deepspeed examples/pytorch/translation/run_translation.py --model_name_or_path t5-3b \
--output_dir ckpt/t5_3b_w_m_conti --overwrite_output_dir \
--source_lang input --target_lang output \
--train_file /home/zeyi/longtail/longtail_data/for_finetune/conti/w_m_t5/train/data.json --test_file /home/zeyi/longtail/longtail_data/for_finetune/conti/w_m_t5/test/data.json --validation_file /home/zeyi/longtail/longtail_data/for_finetune/conti/w_m_t5/dev/data.json \
--max_source_length 256 --max_target_length 256 --val_max_target_length 256 \
--do_train --num_train_epochs 4 --per_device_train_batch_size 8 --learning_rate 1e-4 \
--deepspeed tests/deepspeed/ds_config_zero3.json --save_strategy epoch --evaluation_strategy epoch \
--load_best_model_at_end --warmup_ratio 0.1 --adam_epsilon 1e-6 --weight_decay 0.01


# property-centric 3b
rm -r ckpt/t5_3b_w_m_property_centric; PYTHONPATH=src USE_TF=0 deepspeed examples/pytorch/translation/run_translation.py --model_name_or_path t5-3b \
--output_dir ckpt/t5_3b_w_m_property_centric --overwrite_output_dir \
--source_lang input --target_lang output \
--train_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/w_m_t5/train/data.json --test_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/w_m_t5/test/data.json --validation_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/w_m_t5/dev/data.json \
--max_source_length 256 --max_target_length 256 --val_max_target_length 256 \
--do_train --num_train_epochs 4 --per_device_train_batch_size 8 --learning_rate 1e-4 \
--deepspeed tests/deepspeed/ds_config_zero3.json --save_strategy epoch --evaluation_strategy epoch \
--load_best_model_at_end --warmup_ratio 0.1 --adam_epsilon 1e-6 --weight_decay 0.01


# property-centric 11b
rm -r ckpt/t5_11b_w_m_property_centric; PYTHONPATH=src USE_TF=0 deepspeed examples/pytorch/translation/run_translation.py --model_name_or_path t5-11b \
--output_dir ckpt/t5_11b_w_m_property_centric --overwrite_output_dir \
--source_lang input --target_lang output \
--train_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/w_m_t5/train/data.json --test_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/w_m_t5/test/data.json --validation_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/w_m_t5/dev/data.json \
--max_source_length 256 --max_target_length 256 --val_max_target_length 256 \
--do_train --num_train_epochs 4 --per_device_train_batch_size 8 --learning_rate 1e-4 \
--deepspeed tests/deepspeed/ds_config_zero3.json --save_strategy epoch --evaluation_strategy epoch \
--load_best_model_at_end --warmup_ratio 0.1 --adam_epsilon 1e-6 --weight_decay 0.01





source activate /home/zeyi/miniconda3/envs/longtail
cd /home/zeyi/LongTailedKnowledge/neurologic_decoding/seq2seq


# *****************************
PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m_dis --input_path /home/zeyi/longtail/longtail_data/raw_data/for_dis/inputs_t5_infer.csv --reference_path \
../dataset/clean/commongen.dev.tgt.txt --constraint_file /home/zeyi/longtail/longtail_data/raw_data/for_dis/inflection_constraints_t5_infer.json --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/for_dis/lemma_constraints_t5_infer.json --min_tgt_length 5 \
--max_tgt_length 64 --bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1 \
--early_stop 1 --save_path /home/zeyi/longtail/longtail_data/generated_data/for_dis/t5_3b_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --task constrained_generation --n_obs 200


PYTHONPATH=.. python run_eval.py \
--model_name /home/zeyi/transformers/ckpt/t5_3b_w_m_dis  \
--input_path /home/zeyi/longtail/longtail_data/raw_data/for_dis/inputs_t5_infer.csv --reference_path ../dataset/clean/commongen.dev.tgt.txt  \
--min_tgt_length 5 --max_tgt_length 64 --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/for_dis/lemma_constraints_t5_infer.json \
--bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
--save_path /home/zeyi/longtail/longtail_data/generated_data/for_dis/t5_3b_vanilla_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --n_obs 200


# for conti
PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m_conti --input_path /home/zeyi/longtail/longtail_data/raw_data/conti/inputs_t5_infer.csv --reference_path \
../dataset/clean/commongen.dev.tgt.txt --constraint_file /home/zeyi/longtail/longtail_data/raw_data/conti/inflection_constraints_t5_infer.json --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/conti/lemma_constraints_t5_infer.json --min_tgt_length 5 \
--max_tgt_length 64 --bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1 \
--early_stop 1 --save_path /home/zeyi/longtail/longtail_data/generated_data/conti/t5_3b_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --task constrained_generation --n_obs 200


PYTHONPATH=.. python run_eval.py \
--model_name /home/zeyi/transformers/ckpt/t5_3b_w_m_conti  \
--input_path /home/zeyi/longtail/longtail_data/raw_data/conti/inputs_t5_infer.csv --reference_path ../dataset/clean/commongen.dev.tgt.txt  \
--min_tgt_length 5 --max_tgt_length 64 --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/conti/lemma_constraints_t5_infer.json \
--bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
--save_path /home/zeyi/longtail/longtail_data/generated_data/conti/t5_3b_vanilla_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --n_obs 200



# property centric

PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m_property_centric --input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/inputs_t5_infer.csv --reference_path \
../dataset/clean/commongen.dev.tgt.txt --constraint_file /home/zeyi/longtail/longtail_data/raw_data/property_centric/inflection_constraints_t5_infer.json --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/property_centric/lemma_constraints_t5_infer.json --min_tgt_length 5 \
--max_tgt_length 64 --bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1 \
--early_stop 1 --save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/t5_3b_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --task constrained_generation --n_obs 200


PYTHONPATH=.. python run_eval.py \
--model_name /home/zeyi/transformers/ckpt/t5_3b_w_m_property_centric  \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/inputs_t5_infer.csv --reference_path ../dataset/clean/commongen.dev.tgt.txt  \
--min_tgt_length 5 --max_tgt_length 64 --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/property_centric/lemma_constraints_t5_infer.json \
--bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
--save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/t5_3b_vanilla_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --n_obs 200



# change beam_size to 10

PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/t5_11b_w_m_property_centric --input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/inputs_t5_infer.csv --reference_path \
../dataset/clean/commongen.dev.tgt.txt --constraint_file /home/zeyi/longtail/longtail_data/raw_data/property_centric/inflection_constraints_t5_infer.json --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/property_centric/lemma_constraints_t5_infer.json --min_tgt_length 5 \
--max_tgt_length 64 --bs 2 --beam_size 10 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1 \
--early_stop 1 --save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/t5_11b_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --task constrained_generation --n_obs 200 --parallelize


PYTHONPATH=.. python run_eval.py --model_name /home/zeyi/transformers/ckpt/t5_11b_w_m_property_centric  \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/inputs_t5_infer.csv --reference_path ../dataset/clean/commongen.dev.tgt.txt  \
--min_tgt_length 5 --max_tgt_length 64 --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/property_centric/lemma_constraints_t5_infer.json \
--bs 4 --beam_size 10 --length_penalty 0.1 --ngram_size 3 \
--save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/t5_11b_vanilla_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --n_obs 200 --parallelize

# *****************************




# *****************************
PYTHONPATH=. python GPT_task/GPT_fill_task.py --inputs longtail_data/raw_data/for_dis/inputs_t5_infer.csv \
--lemma_constraints longtail_data/raw_data/for_dis/lemma_constraints_t5_infer.json \
--inflection_constraints longtail_data/raw_data/for_dis/inflection_constraints_t5_infer.json \
--outputs longtail_data/generated_data/for_dis --needed_count 1 --Mturk --num_groups 20 --variations_per_group 8 --no_filter

# for continuation format
PYTHONPATH=. python GPT_task/GPT_fill_task.py --inputs longtail_data/raw_data/conti/inputs_t5_infer.csv \
--lemma_constraints longtail_data/raw_data/conti/lemma_constraints_t5_infer.json \
--inflection_constraints longtail_data/raw_data/conti/inflection_constraints_t5_infer.json \
--outputs longtail_data/generated_data/conti --needed_count 1 --conti --Mturk --num_groups 20 --variations_per_group 8 --no_filter

# property centric
PYTHONPATH=. python GPT_task/GPT_fill_task.py --inputs longtail_data/raw_data/property_centric/inputs_t5_infer.csv \
--lemma_constraints longtail_data/raw_data/property_centric/lemma_constraints_t5_infer.json \
--inflection_constraints longtail_data/raw_data/property_centric/inflection_constraints_t5_infer.json \
--outputs longtail_data/generated_data/property_centric --needed_count 1 --conti --Mturk --num_groups 20 --variations_per_group 8 --no_filter


# *****************************




# *****************************
PYTHONPATH=. python longtail_data/easy_to_read.py --inputs_dir longtail_data/raw_data/for_dis --datas_dir longtail_data/generated_data/for_dis --num_groups 20 --variations_per_group 8


# since we wanna translate back to original templates
PYTHONPATH=. python longtail_data/easy_to_read.py --inputs_dir longtail_data/raw_data/for_dis --datas_dir longtail_data/generated_data/conti --num_groups 20 --variations_per_group 8
# *****************************