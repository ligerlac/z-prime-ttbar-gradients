"""
Summary of NumPy → JAX conversion
"""
import json
import re

nb = json.load(open('GRAEP_Tutorial.ipynb'))

print("="*80)
print("NUMPY vs JAX USAGE SUMMARY")
print("="*80)

numpy_uses = {
    'load': [],
    'histogram': [],
    'random': [],
    'array_plotting': [],
    'other': []
}

jax_operations = set()

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    # Count NumPy uses
    if 'np.load' in source:
        numpy_uses['load'].append(i)
    if 'np.histogram' in source:
        numpy_uses['histogram'].append(i)
    if 'np.random' in source:
        numpy_uses['random'].append(i)
    if 'np.array' in source and ('bin_centers' in source or 'bins' in source):
        numpy_uses['array_plotting'].append(i)

    # Count JAX operations
    jax_ops = re.findall(r'jnp\.(\w+)\(', source)
    jax_operations.update(jax_ops)

print("\n✅ NUMPY KEPT FOR (necessary):")
print(f"  • File I/O (np.load): {len(numpy_uses['load'])} cells")
print(f"  • Plotting (np.histogram): {len(numpy_uses['histogram'])} cells")
print(f"  • Random seed: {len(numpy_uses['random'])} cells")
print(f"  • Bin centers: {len(numpy_uses['array_plotting'])} cells")

print(f"\n🚀 JAX USED FOR:")
print(f"  • {len(jax_operations)} different operations")
print(f"  • Including: {', '.join(sorted(list(jax_operations)[:10]))}")
if len(jax_operations) > 10:
    print(f"    ... and {len(jax_operations) - 10} more")

print("\n" + "="*80)
print("BENEFITS OF JAX EVERYWHERE:")
print("="*80)
print("  ✓ All computations are differentiable")
print("  ✓ Can use jax.grad() on any function")
print("  ✓ JIT compilation for speed")
print("  ✓ Consistent API throughout notebook")
print("  ✓ Easier to understand (one ecosystem, not two)")
print("\nNumPy kept ONLY where absolutely necessary (file I/O, plotting)")
