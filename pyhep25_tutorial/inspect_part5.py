"""
Inspect Part 5 cells to see what's actually there
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

# Find Part 5
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '## Part 5: Statistical Significance with evermore' in source:
        print(f"Part 5 starts at cell {i}\n")
        print("="*80)

        # Show next 10 cells
        for j in range(i, min(i + 10, len(nb['cells']))):
            cell_source = ''.join(nb['cells'][j]['source']) if isinstance(nb['cells'][j]['source'], list) else nb['cells'][j]['source']
            first_line = cell_source.split('\n')[0] if cell_source else "(empty)"

            # Check for key patterns
            has_data_conversion = 'wjets_jax = {' in cell_source
            has_Z_initial = 'Z_initial = ' in cell_source
            has_grad = 'grad_fn = jax.grad' in cell_source

            markers = []
            if has_data_conversion:
                markers.append('[DATA_CONV]')
            if has_Z_initial:
                markers.append('[Z_INIT]')
            if has_grad:
                markers.append('[GRAD]')

            marker_str = ' '.join(markers) if markers else ''

            print(f"Cell {j:2d} ({nb['cells'][j]['cell_type']:8s}): {first_line[:60]:60s} {marker_str}")

        break
