import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

print(f"Total cells: {len(nb['cells'])}\n")
print("PART HEADERS:")
print("="*70)

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '## Part' in source:
        first_line = source.split('\n')[0]
        print(f"Cell {i:2d}: {first_line}")

print("\n" + "="*70)
print("\nRefactoring complete! Ready for testing.")
