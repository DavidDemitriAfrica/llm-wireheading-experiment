import json

with open('wandb_detailed_results.json', 'r') as f:
    data = json.load(f)

print(f'Total entries: {len(data)}')
print('\nAll keys and models:')
for k in sorted(data.keys()):
    model = data[k].get('model', 'unknown')
    print(f'  {k}: {model}')
