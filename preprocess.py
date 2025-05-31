import json
import pprint
import os

num_sample = 1000

input_file = './data/OpenKPEval.jsonl'
suffix = f'{num_sample}' if num_sample > 0 else ''
output_file = f"./data/data{suffix}.json"

clean_data = []
global_idx = 0
done = False


with open(input_file, 'r') as f:
    for idx, line in enumerate(f):
        obj = json.loads(line)
        # print only the "text" field (or a blank string if it's missing)
        # pprint.pprint(obj.get('VDOM', ''))
        vdom_entries = json.loads(obj['VDOM'])
        for entry in vdom_entries:
            # pprint.pprint(entry)
            # for sub_entry in 
            # pprint.pprint(entry['text'])
            # pprint.pprint(entry['feature'])
            entry = {'id': global_idx, 'feature': entry['feature'], 'text': entry['text']}
            clean_data.append(entry)
            
            # print("\n")
        
            global_idx += 1
            if global_idx >= num_sample:
                done = True
                break
        if done:
            break

with open(output_file, 'w') as fout:
    json.dump(clean_data, fout, indent=2)
print(f"Wrote {len(clean_data)} records to {output_file}")
