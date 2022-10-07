

cd seq2seq
source activate hug
# finetuned_ckpt and with mask in inputs --t5-3b
# note: since we trained t5-3b on V4 transformer and seems some layer of it don't match the layer at V3
PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m/checkpoint-18 --input_path /home/zeyi/longtail/longtail_data/raw_data/generation_input_t5_infer.txt --reference_path\
 ../dataset/clean/commongen.1.tgt.txt --constraint_file /home/zeyi/longtail/longtail_data/raw_data/generation_constraints_inflections_t5_infer.json --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/generation_constraints_lemmas_t5_infer.json --min_tgt_length 5\
  --max_tgt_length 64 --bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1\
   --early_stop 1 --save_path /home/zeyi/longtail/longtail_data/generated_data/t5_3b_w_m.txt --score_path ../output_dir/output_file_t5_3b.json --task constrained_generation


# vanilla t5-3b with mask
PYTHONPATH=.. python run_eval.py \
  --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m/checkpoint-18  \
  --input_path /home/zeyi/longtail/longtail_data/raw_data/generation_input_t5_infer.txt --reference_path ../dataset/clean/commongen.1.tgt.txt  \
  --min_tgt_length 5 --max_tgt_length 64 --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/generation_constraints_lemmas_t5_infer.json \
  --bs 16 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
  --save_path /home/zeyi/longtail/longtail_data/generated_data/t5_3b_vanilla_w_m.txt --score_path ../output_dir/output_file_t5_3b.json







# for gpt-generation
cd GPT_task
source activate longtail
PYTHONPATH=.. python GPT_fill_task.py --input /home/zeyi/longtail/longtail_data/raw_data/generation_input_t5_infer.txt --constraint_lemma \
/home/zeyi/longtail/longtail_data/raw_data/generation_constraints_lemmas_t5_infer.json --output /home/zeyi/longtail/longtail_data/generated_data --no_filter --Mturk \
--num_groups 20
