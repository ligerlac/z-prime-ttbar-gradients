"""
Fix Part 5 cell order: data conversion must come before gradient computation
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

# Find Part 5 cells
part5_idx = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '## Part 5: Statistical Significance with evermore' in source:
        part5_idx = i
        break

if part5_idx:
    print(f"Found Part 5 at cell {part5_idx}")

    # Find the cell that calls compute_evermore_significance (has data conversion)
    compute_sig_cell_idx = None
    grad_cell_idx = None

    for i in range(part5_idx + 1, min(part5_idx + 15, len(nb['cells']))):
        source = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']

        if 'Z_initial = compute_evermore_significance' in source:
            compute_sig_cell_idx = i
            print(f"  Found significance computation at cell {i}")

        if 'grad_fn = jax.grad' in source and 'compute_evermore_significance' in source:
            grad_cell_idx = i
            print(f"  Found gradient cell at cell {i}")

    # If gradient cell comes before compute_sig_cell, we have a problem
    if grad_cell_idx and compute_sig_cell_idx and grad_cell_idx < compute_sig_cell_idx:
        print(f"\n⚠ ERROR: Gradient cell ({grad_cell_idx}) comes before data conversion ({compute_sig_cell_idx})")
        print("  Fixing by combining into single cell...")

        # Get both cells
        compute_cell = nb['cells'][compute_sig_cell_idx]
        grad_cell = nb['cells'][grad_cell_idx]

        # Combine them
        compute_source = ''.join(compute_cell['source']) if isinstance(compute_cell['source'], list) else compute_cell['source']
        grad_source = ''.join(grad_cell['source']) if isinstance(grad_cell['source'], list) else grad_cell['source']

        # Create combined cell
        combined_source = compute_source + "\n\n" + grad_source

        # Update the first cell with combined content
        nb['cells'][compute_sig_cell_idx]['source'] = combined_source

        # Delete the second cell
        del nb['cells'][grad_cell_idx]

        print(f"  ✓ Combined cells {compute_sig_cell_idx} and {grad_cell_idx}")
        print(f"  ✓ Deleted cell {grad_cell_idx}")

    elif grad_cell_idx and compute_sig_cell_idx:
        print(f"\n✓ Cell order is correct: compute ({compute_sig_cell_idx}) before gradient ({grad_cell_idx})")
    else:
        print(f"\n⚠ Could not find both cells")

# Save
with open('GRAEP_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("\n✓ Notebook saved")
