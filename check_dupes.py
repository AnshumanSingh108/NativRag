import json
from pathlib import Path
from collections import Counter

files = ['bangladesh_train.json', 'india_train.json', 'qatar_train.json']
all_ids = []

for f in files:
    path = Path('processed_data') / f
    with open(path, 'r', encoding='utf-8') as file:
        docs = json.load(file)

    ids = [f"{d['metadata']['source']}_{d['metadata']['data_id']}" for d in docs]
    all_ids.extend(ids)

    # Check within-source duplicates
    counter = Counter([d['metadata']['data_id'] for d in docs])
    dupes = {k: v for k, v in counter.items() if v > 1}
    if dupes:
        print(f'{f}: {len(dupes)} duplicate data_ids')
        for k, v in list(dupes.items())[:5]:
            print(f'  {k} appears {v} times')
    else:
        print(f'{f}: no duplicates ({len(docs)} docs)')

# Check cross-source duplicates
counter = Counter(all_ids)
cross_dupes = {k: v for k, v in counter.items() if v > 1}
if cross_dupes:
    print(f'\nCross-source duplicates: {len(cross_dupes)}')
    for k, v in list(cross_dupes.items())[:5]:
        print(f'  {k} appears {v} times')
else:
    print(f'\nNo cross-source duplicates. Total unique IDs: {len(all_ids)}')
