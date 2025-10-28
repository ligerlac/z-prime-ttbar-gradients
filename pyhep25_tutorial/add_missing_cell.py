"""
Add missing cell in Part 5 that converts data and computes Z_initial
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

# Find cell 18 (compute_evermore_significance definition)
insert_after_idx = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'def compute_evermore_significance(opt_params' in source:
        insert_after_idx = i
        print(f"Found compute_evermore_significance definition at cell {i}")
        break

if insert_after_idx:
    # Insert new cell after the function definition
    insert_idx = insert_after_idx + 1

    new_cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": """# Convert data to JAX arrays
wjets_jax = {k: jnp.array(v) for k, v in wjets_data.items()}
ttbar_jax = {k: jnp.array(v) for k, v in ttbar_data.items()}
signal_jax = {k: jnp.array(v) for k, v in signal_data.items()}
data_jax = {k: jnp.array(v) for k, v in data_data.items()}

console.print("[cyan]Computing initial significance with evermore...[/cyan]")

# Compute significance with initial parameters
# This function is differentiable - we'll show gradients next!
Z_initial = compute_evermore_significance(initial_params, wjets_jax, ttbar_jax, signal_jax, data_jax)

console.print(f"\\n[yellow]Initial significance: {Z_initial:.2f}σ[/yellow]")
console.print(f"  MET cut: {initial_params['met_cut']:.1f} GeV")
console.print(f"  b-tag cut: {initial_params['btag_cut']:.2f}")
console.print(f"  HT cut: {initial_params['ht_cut']:.1f} GeV")
console.print(f"  Bandwidth: {initial_params['bandwidth']:.1f} GeV")"""
    }

    nb['cells'].insert(insert_idx, new_cell)
    print(f"✓ Inserted data conversion + Z_initial computation at cell {insert_idx}")

# Save
with open('GRAEP_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("\n✓ Fixed! Data conversion now comes before gradient computation")
