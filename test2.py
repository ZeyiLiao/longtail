from datasets import load_dataset
ds = load_dataset('json', data_files='/home/zeyi/longtail/longtail_data/for_finetune/property_centric/wo_m_t5/train/data.json')
print(ds)