


# for gpt-generation
cd GPT_task
source activate longtail
PYTHONPATH=.. python GPT_fill_task.py --input /home/zeyi/longtail/longtail_data/raw_data/generation_input_t5_infer.txt \
--lemma_constraints /home/zeyi/longtail/longtail_data/raw_data/lemma_constraints_t5_infer.json \
--inflection_constraints /home/zeyi/longtail/longtail_data/raw_data/inflection_constraints_t5_infer.json \
--output /home/zeyi/longtail/longtail_data/generated_data --Mturk \
--num_groups 20 --variations_per_group 32 --no_filter


PYTHONPATH=.. python GPT_fill_task.py --input /home/zeyi/longtail/longtail_data/raw_data/for_dis/inputs_t5_infer.txt \
--lemma_constraints /home/zeyi/longtail/longtail_data/raw_data/for_dis/lemma_constraints_t5_infer.json \
--inflection_constraints /home/zeyi/longtail/longtail_data/raw_data/for_dis/inflection_constraints_t5_infer.json \
--output /home/zeyi/longtail/longtail_data/generated_data/for_dis --Mturk \
--num_groups 2 --variations_per_group 8 --no_filter


# for easy to read
PYTHONPATH=.. python easy_to_read.py --inputs_dir /home/zeyi/longtail/longtail_data/raw_data/for_dis \
--datas_dir /home/zeyi/longtail/longtail_data/generated_data/for_dis \
--num_groups 20 --variations_per_group 8
