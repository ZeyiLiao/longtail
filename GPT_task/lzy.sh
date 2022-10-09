
# 1
PYTHONPATH=.. python GPT_fill_task.py --inputs ../longtail_data/raw_data/inputs_t5_train.txt \
--lemma_constraints ../longtail_data/raw_data/lemma_constraints_t5_train.json \
--inflection_constraints ../longtail_data/raw_data/inflection_constraints_t5_train.json \
--output ../longtail_data/for_finetune

# for dis(disjunction constraints)
PYTHONPATH=.. python GPT_fill_task.py --inputs ../longtail_data/raw_data/for_dis/inputs_t5_train.txt \
--lemma_constraints ../longtail_data/raw_data/for_dis/lemma_constraints_t5_train.json \
--inflection_constraints ../longtail_data/raw_data/for_dis/inflection_constraints_t5_train.json \
--output ../longtail_data/for_finetune/for_dis


# 2 splitting

# 3 change into t5 format

# 4 change into tf_trainer format