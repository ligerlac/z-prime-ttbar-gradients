"""
Fix build_hist_with_cuts function in Part 4 - it seems to be missing or commented out
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

# Find cell 15 and check what's there
cell_15_source = ''.join(nb['cells'][15]['source']) if isinstance(nb['cells'][15]['source'], list) else nb['cells'][15]['source']

print("Cell 15 current content:")
print("="*80)
print(cell_15_source[:200])
print("="*80)

# If it's just a comment, replace with actual function
if '# This function is defined in Part 4' in cell_15_source:
    print("\n⚠ Found placeholder comment, replacing with actual function...")

    nb['cells'][15]['source'] = """# Combining soft cuts + binned KDE into one building block
def build_hist_with_cuts(data_dict, opt_params, bin_edges, temperature=5.0):
    \"\"\"
    Build binned KDE histogram with soft cuts applied.

    This combines two building blocks:
    - Soft selection (from Part 3)
    - Binned KDE (from Part 4)

    Parameters
    ----------
    data_dict : dict
        Data dictionary with JAX arrays
    opt_params : dict
        Optimization parameters:
        - met_cut, btag_cut, ht_cut (selection cuts)
        - bandwidth (histogram smoothing)
    bin_edges : array
        Histogram bin edges
    temperature : float
        Soft cut temperature

    Returns
    -------
    hist : array
        Differentiable histogram
    \"\"\"
    # Extract m_ttbar values and weights (already JAX arrays from Part 2!)
    values = data_dict['m_ttbar']
    weights = data_dict['weight']

    # Apply soft selection (uses cut parameters)
    met_pass = jax.nn.sigmoid((data_dict['met_pt'] - opt_params['met_cut']) / temperature)
    btag_pass = jax.nn.sigmoid((data_dict['leading_jet_btag'] - opt_params['btag_cut']) / temperature)
    ht_pass = jax.nn.sigmoid((data_dict['st'] - opt_params['ht_cut']) / temperature)
    selection_weights = met_pass * btag_pass * ht_pass

    final_weights = weights * selection_weights

    # Binned KDE (uses bandwidth parameter)
    bandwidth = opt_params['bandwidth']

    # Use binned_kde_histogram from earlier in Part 4
    histogram = binned_kde_histogram(values, final_weights, bin_edges, bandwidth)

    return histogram

console.print("[green]✓ Building block defined:[/green] build_hist_with_cuts()")
console.print("  Combines soft cuts + binned KDE")
console.print("  Works with JAX arrays from Part 2")
console.print("  Differentiable w.r.t. ALL parameters (cuts + bandwidth)")"""

    print("✓ Replaced placeholder with actual function")

# Save
with open('GRAEP_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("\n✓ build_hist_with_cuts function now properly defined in Part 4")
