"""
Fix typo: jjnp → jnp (double j was introduced by regex replacement)
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

fixes = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    if 'jjnp' in source:
        nb['cells'][i]['source'] = source.replace('jjnp', 'jnp')
        fixes += 1
        print(f"✓ Fixed cell {i}: jjnp → jnp")

    # Also fix any jsignal_data → signal_data typos
    if 'jsignal_data' in source:
        nb['cells'][i]['source'] = nb['cells'][i]['source'].replace('jsignal_data', 'signal_data')
        print(f"✓ Fixed cell {i}: jsignal_data → signal_data")

with open('GRAEP_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print(f"\n✓ Fixed {fixes} cells with jjnp typo")
