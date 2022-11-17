#!/bin/bash

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

# **********************************************************


source activate /home/zeyi/miniconda3/envs/longtail
cd /home/zeyi/neurologic_decoding_lzy/seq2seq


# *****************************


# property centric
# disable early stopping, increse beta, increase alpha,   incerase the search space

# add a argument named change_format inorder to have different format


PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/w_m_t5_3b_property_centric \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--min_tgt_length 5 --max_tgt_length 128 --bs 2 --beam_size 10 --length_penalty 0.1 --ngram_size 3 --prune_factor 500000 --beta 2 \
--early_stop 0 --save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/w_m_t5_3b.csv --n_obs 200 --parallelize


PYTHONPATH=.. python run_eval.py \
--model_name /home/zeyi/transformers/ckpt/w_m_t5_3b_property_centric \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--min_tgt_length 5 --max_tgt_length 128 \
--bs 8 --beam_size 10 --length_penalty 0.1 --ngram_size 3 \
--save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/w_m_t5_3b_vanilla.csv --n_obs 200 --parallelize


PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/wo_m_t5_3b_property_centric \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--min_tgt_length 5 --max_tgt_length 128 --bs 2 --beam_size 10 --length_penalty 0.1 --ngram_size 3 --prune_factor 500000 --beta 2 \
--early_stop 0 --save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_t5_3b.csv --n_obs 200 --parallelize --wo_mask


PYTHONPATH=.. python run_eval.py \
--model_name /home/zeyi/transformers/ckpt/wo_m_t5_3b_property_centric  \
--input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
--min_tgt_length 5 --max_tgt_length 128 \
--bs 8 --beam_size 10 --length_penalty 0.1 --ngram_size 3 \
--save_path /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_t5_3b_vanilla.csv --n_obs 200 --parallelize --wo_mask


# source activate hug    since it's v3 transformer
python decode_pt.py --model_name /home/zeyi/finetune/saved/lrgenerative_gpt2_large_wo_m_gpt2_16_11_2022_22db49ec/checkpoints/pytorch_model.pkl \
  --output_file /home/zeyi/longtail/longtail_data/generated_data/property_centric/wo_m_gpt2_large.csv \
  --input_path /home/zeyi/longtail/longtail_data/raw_data/property_centric/infer_ids.csv \
  --batch_size 8 --beam_size 10 --max_tgt_length 128 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.1 \
  --prune_factor 500000 --beta 2 --early_stop 10 --n_obs 200








# *****************************


# property centric
python GPT_fill_task.py --inputs ../longtail_data/raw_data/property_centric/infer_ids.csv \
--outputs ../longtail_data/generated_data/property_centric --needed_count 1 --conti --no_filter --n_obs 40


# *****************************




# *****************************
python easy_to_read.py --dir ./generated_data/property_centric


# *****************************