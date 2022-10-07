from collections import defaultdict as ddict
from pathlib import Path
import json

l = []
with open('/home/zeyi/longtail/longtail_data/for_finetune/w_m_t5/test/data.json') as f:
    for line in f:
        data = json.loads(line)
        l.append(data)

print(1)