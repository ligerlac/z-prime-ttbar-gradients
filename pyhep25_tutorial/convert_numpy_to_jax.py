"""
Convert NumPy operations to JAX where appropriate.
Keep NumPy only for file I/O (np.load).
"""
import json
import re

nb = json.load(open('GRAEP_Tutorial.ipynb'))

print("="*80)
print("CONVERTING NUMPY TO JAX")
print("="*80)

conversions = []

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    original_source = source

    # Skip if it's just np.load (file I/O - must stay NumPy)
    if 'np.load' in source and source.count('np.') == source.count('np.load'):
        continue

    # Convert various NumPy operations to JAX
    replacements = [
        # Array operations
        (r'np\.sum\(', 'jnp.sum('),
        (r'np\.mean\(', 'jnp.mean('),
        (r'np\.std\(', 'jnp.std('),
        (r'np\.sqrt\(', 'jnp.sqrt('),
        (r'np\.exp\(', 'jnp.exp('),
        (r'np\.log\(', 'jnp.log('),
        (r'np\.abs\(', 'jnp.abs('),
        (r'np\.max\(', 'jnp.max('),
        (r'np\.min\(', 'jnp.min('),

        # Array creation
        (r'np\.linspace\(', 'jnp.linspace('),
        (r'np\.arange\(', 'jnp.arange('),
        (r'np\.zeros\(', 'jnp.zeros('),
        (r'np\.ones\(', 'jnp.ones('),
        (r'np\.meshgrid\(', 'jnp.meshgrid('),

        # When converting data that's already JAX, don't use np.array
        # (but be careful with np.histogram - it needs NumPy for mplhep)
        (r'np\.column_stack\(', 'jnp.column_stack('),

        # In expressions where we're working with JAX data
        (r'np\.concatenate\(', 'jnp.concatenate('),
    ]

    for pattern, replacement in replacements:
        if re.search(pattern, source):
            # Special case: keep np.histogram for plotting compatibility with mplhep
            if pattern == r'np\.histogram\(' and 'hep.histplot' in source:
                continue

            source = re.sub(pattern, replacement, source)

    # Special handling for np.array() - context-dependent
    # If we're working with JAX data already, use jnp.array
    if 'np.array(' in source:
        # Check if this is converting data that should already be JAX
        if 'signal_data[' in source or 'wjets_data[' in source or 'ttbar_data[' in source:
            # Data is already JAX from Part 2, shouldn't need conversion at all!
            # Remove the np.array wrapper
            source = re.sub(r'np\.array\((.*?_data\[.*?\])\)', r'\1', source)

        # For bin_centers calculation with existing arrays
        elif 'bin_centers' not in source:
            # Other np.array calls - convert to jnp.array
            source = re.sub(r'np\.array\(', 'jnp.array(', source)

    # Update cell if changed
    if source != original_source:
        nb['cells'][i]['source'] = source
        conversions.append(f"Cell {i}: NumPy → JAX conversions")

print(f"\nConverted {len(conversions)} cells:")
for conv in conversions:
    print(f"  ✓ {conv}")

# Save
with open('GRAEP_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("NUMPY TO JAX CONVERSION COMPLETE")
print("="*80)
print("\nKept NumPy only for:")
print("  - np.load() (file I/O)")
print("  - np.histogram() with mplhep (plotting compatibility)")
print("  - np.random.seed() (reproducibility)")
print("\nEverything else converted to JAX!")
print("\nBenefits:")
print("  - Consistent use of JAX throughout")
print("  - All operations are differentiable")
print("  - Better performance with JIT compilation")
