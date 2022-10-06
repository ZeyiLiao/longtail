

# for gpt-generation
cd GPT_task

PYTHONPATH=.. python GPT_fill_task.py --input /home/zeyi/longtail/Mturk_data/t5_input_w_mask.txt --constraint_lemma \
/home/zeyi/longtail/Mturk_data/constraint_lemmas.json --output /home/zeyi/longtail/Mturk_data/GPT_output_w_m.txt --no_filter --Mturk
