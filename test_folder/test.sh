#!/usr/bin/env bash

uuid="$(uuidgen)"
model="gpt2-large"
data_type=$1

python run_clm.py --model_name_or_path $model --output_dir "./$model-$uuid" \
--train_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/$data_type \
--validation_file /home/zeyi/longtail/longtail_data/for_finetune/property_centric/$data_type \





