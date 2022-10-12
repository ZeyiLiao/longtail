#!/bin/bash

source activate /home/zeyi/miniconda3/envs/longtail

PYTHONPATH=. python raw_generation_data/prepare_raw_data.py --input_file longtail_data/input.csv --save_file longtail_data/related_words.pkl \
--csv_file longtail_data/ATOMIC10X_filter.csv --output_file longtail_data

PYTHONPATH=. python GPT_task/GPT_fill_task.py --inputs longtail_data/raw_data/for_dis/inputs_t5_train.csv \
--lemma_constraints longtail_data/raw_data/for_dis/lemma_constraints_t5_train.json \
--inflection_constraints longtail_data/raw_data/for_dis/inflection_constraints_t5_train.json \
--outputs longtail_data/for_finetune/for_dis

PYTHONPATH=. python longtail_data/splitting.py --gpt_outputs_dir longtail_data/for_finetune/for_dis


source activate /home/zeyi/miniconda3/envs/tf_trainer

cd /home/zeyi/transformers

rm -r t5_3b_w_m_dis; PYTHONPATH=src USE_TF=0 deepspeed examples/pytorch/translation/run_translation.py --model_name_or_path t5-3b \
--output_dir ckpt/t5_3b_w_m_dis --overwrite_output_dir \
--source_lang input --target_lang output \
--train_file /home/zeyi/longtail/longtail_data/for_finetune/for_dis/w_m_t5/train/data.json --test_file /home/zeyi/longtail/longtail_data/for_finetune/for_dis/w_m_t5/test/data.json --validation_file /home/zeyi/longtail/longtail_data/for_finetune/for_dis/w_m_t5/dev/data.json \
--max_source_length 256 --max_target_length 256 --val_max_target_length 256 \
--do_train --num_train_epochs 4 --per_device_train_batch_size 8 --learning_rate 1e-4 \
--deepspeed tests/deepspeed/ds_config_zero3.json --save_strategy epoch --evaluation_strategy epoch \
--load_best_model_at_end --warmup_ratio 0.1 --adam_epsilon 1e-6 --weight_decay 0.01



source activate /home/zeyi/miniconda3/envs/hug
cd /home/zeyi/neurologic_decoding/seq2seq

PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m_dis --input_path /home/zeyi/longtail/longtail_data/raw_data/for_dis/inputs_t5_infer.csv --reference_path \
../dataset/clean/commongen.1.tgt.txt --constraint_file /home/zeyi/longtail/longtail_data/raw_data/for_dis/inflection_constraints_t5_infer.json --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/for_dis/lemma_constraints_t5_infer.json --min_tgt_length 5 \
--max_tgt_length 64 --bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1 \
--early_stop 1 --save_path /home/zeyi/longtail/longtail_data/generated_data/for_dis/t5_3b_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --task constrained_generation


PYTHONPATH=.. python run_eval.py \
--model_name /home/zeyi/transformers/ckpt/t5_3b_w_m_dis  \
--input_path /home/zeyi/longtail/longtail_data/raw_data/for_dis/inputs_t5_infer.csv --reference_path ../dataset/clean/commongen.1.tgt.txt  \
--min_tgt_length 5 --max_tgt_length 64 --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/for_dis/lemma_constraints_t5_infer.json \
--bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
--save_path /home/zeyi/longtail/longtail_data/generated_data/for_dis/t5_3b_vanilla_w_m.csv --score_path ../output_dir/output_file_t5_3b.json
