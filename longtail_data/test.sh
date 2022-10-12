#!/bin/bash

source activate /home/zeyi/miniconda3/envs/longtail

# export PYTHONPATH=.
# python raw_generation_data/prepare_raw_data.py --input_file longtail_data/input.csv --save_file longtail_data/related_words.pkl \
# --csv_file longtail_data/ATOMIC10X_filter.csv --output_file longtail_data

# PYTHONPATH=. python GPT_task/GPT_fill_task.py --inputs longtail_data/raw_data/for_dis/inputs_t5_train.csv \
# --lemma_constraints longtail_data/raw_data/for_dis/lemma_constraints_t5_train.json \
# --inflection_constraints longtail_data/raw_data/for_dis/inflection_constraints_t5_train.json \
# --output longtail_data/for_finetune/for_dis

# python longtail_data/splitting.py --gpt_outputs_dir longtail_data/for_finetune/for_dis
# echo $PYTHONPATH
