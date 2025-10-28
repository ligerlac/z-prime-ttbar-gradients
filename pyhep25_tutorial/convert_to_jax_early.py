"""
Refactor to convert data to JAX arrays right after loading in Part 2,
then use JAX arrays everywhere (remove repeated conversions)
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

print("="*80)
print("CONVERTING TO JAX EARLY")
print("="*80)

# ============================================================================
# PART 2: Add JAX conversion after data loading
# ============================================================================

part2_idx = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '## Part 2: Load and Explore Data' in source:
        part2_idx = i
        print(f"✓ Found Part 2 at cell {i}")
        break

if part2_idx:
    # Find the cell that loads the data
    for i in range(part2_idx + 1, min(part2_idx + 15, len(nb['cells']))):
        source = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']

        # Look for data loading
        if 'np.load' in source and 'tutorial_data' in source:
            # Add JAX conversion cell right after this one
            insert_idx = i + 1

            new_cell = {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": """# Convert all data to JAX arrays (do this ONCE at the start)
# From now on, we work with JAX arrays everywhere!

def convert_to_jax(data_dict):
    \"\"\"Convert all arrays in dictionary to JAX arrays.\"\"\"
    return {k: jnp.array(v) for k, v in data_dict.items()}

wjets_data = convert_to_jax(wjets_data)
ttbar_data = convert_to_jax(ttbar_data)
signal_data = convert_to_jax(signal_data)
data_data = convert_to_jax(data_data)

console.print("[green]✓ All data converted to JAX arrays[/green]")
console.print("  From now on, we work with JAX (not NumPy)")"""
            }

            nb['cells'].insert(insert_idx, new_cell)
            print(f"✓ Added JAX conversion at cell {insert_idx}")
            break

# ============================================================================
# PART 4: Update build_hist_with_cuts to expect JAX arrays
# ============================================================================

print("\n" + "="*80)
print("UPDATING FUNCTIONS TO USE JAX ARRAYS")
print("="*80)

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    # Update build_hist_with_cuts
    if 'def build_hist_with_cuts' in source:
        # Replace jnp.array() conversions with direct access
        updated_source = source.replace(
            "values = jnp.array(data_dict['m_ttbar'])",
            "values = data_dict['m_ttbar']  # Already JAX array!"
        ).replace(
            "weights = jnp.array(data_dict['weight'])",
            "weights = data_dict['weight']  # Already JAX array!"
        )

        nb['cells'][i]['source'] = updated_source
        print(f"✓ Updated build_hist_with_cuts at cell {i}")
        break

# ============================================================================
# PART 5: Remove redundant JAX conversions
# ============================================================================

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    # Remove the "Convert data to JAX arrays" cell (now redundant)
    if '# Convert data to JAX arrays' in source and 'wjets_jax = {' in source:
        # Replace with simpler version that just references the already-converted data
        nb['cells'][i]['source'] = """# Data is already in JAX format (converted in Part 2)
# Just create references for clarity
wjets_jax = wjets_data
ttbar_jax = ttbar_data
signal_jax = signal_data
data_jax = data_data

console.print("[cyan]Computing initial significance with evermore...[/cyan]")

# Compute significance with initial parameters
Z_initial = compute_evermore_significance(initial_params, wjets_jax, ttbar_jax, signal_jax, data_jax)

console.print(f"\\n[yellow]Initial significance: {Z_initial:.2f}σ[/yellow]")
console.print(f"  MET cut: {initial_params['met_cut']:.1f} GeV")
console.print(f"  b-tag cut: {initial_params['btag_cut']:.2f}")
console.print(f"  HT cut: {initial_params['ht_cut']:.1f} GeV")
console.print(f"  Bandwidth: {initial_params['bandwidth']:.1f} GeV")"""

        print(f"✓ Updated Part 5 data references at cell {i}")
        break

# ============================================================================
# Update any other jnp.array() conversions in building blocks
# ============================================================================

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    # Check for any remaining jnp.array(data_dict[...]) patterns
    if 'jnp.array(data_dict[' in source or "jnp.array(data_dict['" in source:
        # Replace with direct access
        updated_source = source.replace("jnp.array(data_dict['", "data_dict['")
        updated_source = updated_source.replace('jnp.array(data_dict["', 'data_dict["')

        # Only update if something changed
        if updated_source != source:
            nb['cells'][i]['source'] = updated_source
            print(f"✓ Removed redundant jnp.array() conversions at cell {i}")

print("\nSaving updated notebook...")
with open('GRAEP_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("JAX CONVERSION REFACTORING COMPLETE")
print("="*80)
print("✓ Data converted to JAX in Part 2")
print("✓ All functions now expect JAX arrays")
print("✓ Removed redundant conversions throughout")
print("\nBenefits:")
print("  - Cleaner code (convert once, not repeatedly)")
print("  - More consistent (always JAX, not mixed)")
print("  - More realistic (matches real workflows)")
