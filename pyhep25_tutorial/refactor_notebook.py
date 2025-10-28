"""
Comprehensive notebook refactoring:
1. Readability improvements (pedagogical headers, comments)
2. Technical changes (bandwidth param, renaming, bug fixes)
3. Remove function redefinitions
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

print("="*80)
print("GRAEP TUTORIAL REFACTORING")
print("="*80)

# ============================================================================
# PART 3: Add pedagogical header
# ============================================================================

part3_header_idx = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '## Part 3: Differentiable Cuts' in source:
        part3_header_idx = i
        break

if part3_header_idx:
    nb['cells'][part3_header_idx]['source'] = """## Part 3: Differentiable Cuts

**What you know**: Selection cuts filter events (e.g., MET > 100 GeV → accept/reject)

**What's new**: Soft cuts use sigmoid functions → smooth transition from 0 to 1

**Why it matters**: Smooth = differentiable = we can compute ∂(efficiency)/∂(cut threshold)!

### Soft Selection with Sigmoid

Traditional HEP cuts are **hard**: event either passes (weight=1) or fails (weight=0).

For optimization, we need **soft cuts**: smooth transitions using sigmoid functions.

$$
w_{\\text{soft}} = \\sigma\\left(\\frac{x - \\text{threshold}}{T}\\right) = \\frac{1}{1 + e^{-(x - \\text{threshold})/T}}
$$

where $T$ is the temperature (controls smoothness)."""

print("✓ Part 3: Added pedagogical header")

# ============================================================================
# PART 4: Add pedagogical header + build_hist_with_cuts function
# ============================================================================

part4_header_idx = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '## Part 4: Differentiable Histograms' in source:
        part4_header_idx = i
        break

if part4_header_idx:
    nb['cells'][part4_header_idx]['source'] = """## Part 4: Differentiable Histograms with Binned KDE

**What you know**: Histograms bin data into discrete bins

**What's new**: Binned KDE uses Gaussian CDF → smooth, differentiable histograms

**Why it matters**: We can compute ∂(histogram)/∂(bin edges) AND ∂(histogram)/∂(bandwidth)!

### Binned KDE Formula

Instead of hard binning, we use Gaussian CDFs:

$$
h_i = \\sum_{j=1}^{N} w_j \\left[ \\Phi\\left(\\frac{e_{i+1} - x_j}{\\sigma}\\right) - \\Phi\\left(\\frac{e_i - x_j}{\\sigma}\\right) \\right]
$$

where $\\Phi$ is the Gaussian CDF, $w_j$ are event weights, and $\\sigma$ is the bandwidth.

This is **fully differentiable** w.r.t. both data and bin edges!"""

    # Find the cell after binned_kde_histogram definition to add build_hist_with_cuts
    for i in range(part4_header_idx, len(nb['cells'])):
        source = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']
        if 'def binned_kde_histogram' in source and i < len(nb['cells']) - 1:
            # Check if next cell after gradient demo would be good place
            # Look for the cell showing gradients of histogram
            insert_idx = i + 4  # After binned_kde definition and demos

            # Insert build_hist_with_cuts function
            new_cell = {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": """# Combining soft cuts + binned KDE into one building block
def build_hist_with_cuts(data_dict, opt_params, bin_edges, temperature=5.0):
    \"\"\"
    Build binned KDE histogram with soft cuts applied.

    This combines:
    - Soft selection (from Part 3)
    - Binned KDE (from Part 4)

    Parameters
    ----------
    data_dict : dict
        Data dictionary with features
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
    # Extract m_ttbar values and weights
    values = jnp.array(data_dict['m_ttbar'])
    weights = jnp.array(data_dict['weight'])

    # Apply soft selection (uses cut parameters)
    met_pass = jax.nn.sigmoid((data_dict['met_pt'] - opt_params['met_cut']) / temperature)
    btag_pass = jax.nn.sigmoid((data_dict['leading_jet_btag'] - opt_params['btag_cut']) / temperature)
    ht_pass = jax.nn.sigmoid((data_dict['st'] - opt_params['ht_cut']) / temperature)
    selection_weights = met_pass * btag_pass * ht_pass

    final_weights = weights * selection_weights

    # Binned KDE (uses bandwidth parameter)
    bandwidth = opt_params['bandwidth']
    cdf = jax.scipy.stats.norm.cdf(
        bin_edges.reshape(-1, 1),
        loc=values.reshape(1, -1),
        scale=bandwidth,
    )
    weighted_cdf = cdf * final_weights.reshape(1, -1)
    bin_weights = weighted_cdf[1:, :] - weighted_cdf[:-1, :]
    histogram = jnp.sum(bin_weights, axis=1)

    return histogram

console.print("[green]✓ Building block defined:[/green] build_hist_with_cuts()")
console.print("  Combines soft cuts + binned KDE")
console.print("  Differentiable w.r.t. ALL parameters (cuts + bandwidth)")"""
            }

            nb['cells'].insert(insert_idx, new_cell)
            print(f"✓ Part 4: Added build_hist_with_cuts function at cell {insert_idx}")
            break

    print("✓ Part 4: Added pedagogical header")

# ============================================================================
# PART 5: Major refactoring
# ============================================================================

print("\n" + "="*80)
print("PART 5 REFACTORING")
print("="*80)

part5_header_idx = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '## Part 5: Statistical Significance with evermore' in source:
        part5_header_idx = i
        break

if part5_header_idx:
    # Update header
    nb['cells'][part5_header_idx]['source'] = """## Part 5: Statistical Significance with evermore

**What you know**: Significance measures signal vs background separation (e.g., S/√B)

**What's new**: Profile likelihood accounts for Poisson statistics + nuisance parameters

**Why it matters**: More accurate significance + **it's differentiable**!

### Profile Likelihood Ratio Test

Test statistic for discovery:
$$
q_0 = -2\\ln\\frac{L(\\mu=0, \\hat{\\hat{\\theta}})}{L(\\hat{\\mu}, \\hat{\\theta})}
$$

Significance: $Z = \\Phi^{-1}(1 - p)$ where $p = 1 - \\Phi(\\sqrt{q_0})$

[evermore](https://github.com/pfackeldey/evermore) makes this **fully differentiable** with JAX!

**Key insight**: We can compute $\\frac{\\partial Z}{\\partial \\theta_{\\text{cut}}}$ → enables gradient-based optimization!"""

    print("✓ Part 5: Updated pedagogical header")

    # Find and update the setup cell
    for i in range(part5_header_idx + 1, min(part5_header_idx + 10, len(nb['cells']))):
        source = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']

        # Cell with initial_cuts definition - rename and add bandwidth
        if 'initial_cuts' in source and 'import evermore' in source:
            # Replace this cell entirely
            nb['cells'][i]['source'] = """import evermore as evm
import optimistix as optx

# Initial optimization parameters
# Includes both CUTS (MET, b-tag, HT) and HISTOGRAM settings (bandwidth)
initial_params = {
    'met_cut': jnp.array(50.0),      # MET threshold [GeV]
    'btag_cut': jnp.array(0.5),      # b-tagging score threshold
    'ht_cut': jnp.array(200.0),      # HT threshold [GeV]
    'bandwidth': jnp.array(40.0),    # KDE smoothing parameter [GeV]
}

# Define m_ttbar bins for fit
m_ttbar_bins = jnp.linspace(500, 3500, 15)  # 14 bins in ttbar mass

console.print("[cyan]Initial optimization parameters:[/cyan]")
console.print(f"  MET cut: {initial_params['met_cut']:.1f} GeV")
console.print(f"  b-tag cut: {initial_params['btag_cut']:.2f}")
console.print(f"  HT cut: {initial_params['ht_cut']:.1f} GeV")
console.print(f"  Bandwidth: {initial_params['bandwidth']:.1f} GeV")"""

            print(f"✓ Part 5: Renamed initial_cuts → initial_params, added bandwidth (cell {i})")
            break

    # Fix st_bins → m_ttbar_bins bug and update compute_evermore_significance
    for i in range(part5_header_idx + 1, min(part5_header_idx + 15, len(nb['cells']))):
        source = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']

        if 'def compute_evermore_significance' in source:
            # Replace entire function to use opt_params and m_ttbar_bins
            nb['cells'][i]['source'] = """def compute_evermore_significance(opt_params, wjets_dict, ttbar_dict, signal_dict, data_dict):
    \"\"\"
    Compute discovery significance using evermore profile likelihood.

    This function is FULLY DIFFERENTIABLE w.r.t. opt_params!

    Parameters
    ----------
    opt_params : dict
        Optimization parameters (cuts + bandwidth)
    *_dict : dict
        Data dictionaries for each process

    Returns
    -------
    significance : float
        Discovery significance in σ
    \"\"\"
    # Build histograms with current parameters (uses build_hist_with_cuts from Part 4)
    signal_hist = build_hist_with_cuts(signal_dict, opt_params, m_ttbar_bins)
    ttbar_hist = build_hist_with_cuts(ttbar_dict, opt_params, m_ttbar_bins)
    wjets_hist = build_hist_with_cuts(wjets_dict, opt_params, m_ttbar_bins)
    data_hist = build_hist_with_cuts(data_dict, opt_params, m_ttbar_bins)

    templates = {
        "signal": signal_hist,
        "ttbar": ttbar_hist,
        "wjets": wjets_hist,
    }

    # Define evermore statistical model (Poisson likelihood)
    def statistical_model(params, templates):
        \"\"\"Apply scaling modifiers to templates.\"\"\"
        signal_scaled = params["mu"].scale()(templates["signal"])
        ttbar_scaled = params["scale_ttbar"].scale()(templates["ttbar"])
        wjets_unscaled = templates["wjets"]
        return signal_scaled + ttbar_scaled + wjets_unscaled

    def poisson_nll(params, templates, observation):
        \"\"\"Poisson negative log-likelihood.\"\"\"
        expected = statistical_model(params, templates)
        log_likelihood = evm.pdf.PoissonContinuous(lamb=expected).log_prob(observation).sum()

        # Parameter constraints
        constraints = evm.loss.get_log_probs(params)
        constraints = jax.tree.map(jnp.sum, constraints)
        log_likelihood += evm.util.sum_over_leaves(constraints)

        return -log_likelihood

    # Setup evermore parameters
    evm_params = {
        "mu": evm.Parameter(value=1.0, name="mu"),
        "scale_ttbar": evm.Parameter(value=1.0, name="scale_ttbar"),
    }

    # Unconditional fit (μ floats)
    dynamic, static = evm.tree.partition(evm_params, filter=evm.filter.is_not_frozen)

    def loss_unconditional(dyn):
        p = evm.tree.combine(dyn, static)
        return poisson_nll(p, templates, data_hist)

    solver = optx.BFGS(rtol=1e-4, atol=1e-6)
    result_unconditional = optx.minimise(loss_unconditional, solver, dynamic, max_steps=500)
    nll_unconditional = loss_unconditional(result_unconditional.value)
    bestfit_params = evm.tree.combine(result_unconditional.value, static)
    mu_hat = bestfit_params["mu"].value

    # Conditional fit (μ=0 fixed)
    params_mu0 = {
        "mu": evm.Parameter(value=0.0, name="mu", frozen=True),
        "scale_ttbar": evm.Parameter(value=1.0, name="scale_ttbar"),
    }
    dynamic_mu0, static_mu0 = evm.tree.partition(params_mu0, filter=evm.filter.is_not_frozen)

    def loss_conditional(dyn):
        p = evm.tree.combine(dyn, static_mu0)
        return poisson_nll(p, templates, data_hist)

    result_conditional = optx.minimise(loss_conditional, solver, dynamic_mu0, max_steps=500)
    nll_conditional = loss_conditional(result_conditional.value)

    # Likelihood ratio test
    q0 = 2.0 * (nll_conditional - nll_unconditional)
    q0 = jnp.where(mu_hat >= 0.0, q0, 0.0)

    # Significance
    p_value = 1.0 - jax.scipy.stats.norm.cdf(jnp.sqrt(q0))
    significance = jax.scipy.stats.norm.ppf(1.0 - p_value)

    return significance

console.print("[green]✓ Statistical model defined[/green]")
console.print("  Uses build_hist_with_cuts from Part 4")
console.print("  Fits signal+background to data")
console.print("  Returns discovery significance")"""

            print(f"✓ Part 5: Updated compute_evermore_significance (cell {i})")
            print("  - Uses opt_params instead of cut_params")
            print("  - Fixed st_bins → m_ttbar_bins bug")
            print("  - Hidden evermore internals in function")
            break

    # Update the cell that calls compute_evermore_significance
    for i in range(part5_header_idx + 1, min(part5_header_idx + 20, len(nb['cells']))):
        source = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']

        if 'Z_initial = compute_evermore_significance' in source:
            nb['cells'][i]['source'] = """# Convert data to JAX
wjets_jax = {k: jnp.array(v) for k, v in wjets_data.items()}
ttbar_jax = {k: jnp.array(v) for k, v in ttbar_data.items()}
signal_jax = {k: jnp.array(v) for k, v in signal_data.items()}
data_jax = {k: jnp.array(v) for k, v in data_data.items()}

console.print("[cyan]Computing significance with evermore...[/cyan]")

# The magic: this function is differentiable!
Z_initial = compute_evermore_significance(initial_params, wjets_jax, ttbar_jax, signal_jax, data_jax)

console.print(f"\\n[yellow]Initial significance: {Z_initial:.2f}σ[/yellow]")
console.print(f"  MET cut: {initial_params['met_cut']:.1f} GeV")
console.print(f"  b-tag cut: {initial_params['btag_cut']:.2f}")
console.print(f"  HT cut: {initial_params['ht_cut']:.1f} GeV")
console.print(f"  Bandwidth: {initial_params['bandwidth']:.1f} GeV")"""

            print(f"✓ Part 5: Updated significance computation call (cell {i})")
            break

    # Update gradient demonstration
    for i in range(part5_header_idx + 1, min(part5_header_idx + 20, len(nb['cells']))):
        source = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']

        if 'grad_fn = jax.grad' in source and 'compute_evermore_significance' in source:
            nb['cells'][i]['source'] = """# The KEY MOMENT: Compute gradients of significance!
# Traditional HEP: Try many cut values manually → slow
# GRAEP: Use gradients to find optimal cuts → fast!

grad_fn = jax.grad(lambda params: compute_evermore_significance(params, wjets_jax, ttbar_jax, signal_jax, data_jax))

grads = grad_fn(initial_params)

console.print(f"\\n[cyan]Gradients of significance w.r.t. parameters:[/cyan]")
console.print(f"  ∂Z/∂(MET cut) = {grads['met_cut']:.4f}")
console.print(f"  ∂Z/∂(b-tag cut) = {grads['btag_cut']:.4f}")
console.print(f"  ∂Z/∂(HT cut) = {grads['ht_cut']:.4f}")
console.print(f"  ∂Z/∂(bandwidth) = {grads['bandwidth']:.4f}")
console.print(f"\\n[green]✓ All parameters are differentiable![/green]")
console.print("[dim]Next (Part 6): Use these gradients to optimize...[/dim]")"""

            print(f"✓ Part 5: Updated gradient demonstration (cell {i})")
            break

print("\nSaving refactored notebook...")
with open('GRAEP_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("PHASE 1 COMPLETE: Parts 3, 4, 5 refactored")
print("="*80)
print("✓ Pedagogical headers added")
print("✓ build_hist_with_cuts added to Part 4")
print("✓ Part 5: cut_params → opt_params")
print("✓ Part 5: bandwidth added to parameters")
print("✓ Part 5: st_bins → m_ttbar_bins bug fixed")
print("✓ Part 5: evermore internals hidden in function")
print("\nNext: Part 6 (optimization) and Part 7 (visualizations)")
