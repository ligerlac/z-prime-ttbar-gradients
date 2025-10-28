"""
Update before/after plot to include ratio panel with uncertainties
and display best-fit μ and κ_ttbar from evermore fit.
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

# Find the cell with plot_mttbar_with_fit
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    if 'def plot_mttbar_with_fit' in source:
        print(f"Found plot_mttbar_with_fit in cell {i}")

        # Replace with improved version
        new_source = """# Before/After m_ttbar distribution with evermore fit
console.print("\\n[cyan]Building before/after comparison...[/cyan]")

# Helper function to build and visualize fit with ratio panel
def plot_mttbar_with_fit(fig, gs, opt_params, title, col_idx):
    \"\"\"Plot m_ttbar distribution with evermore fit overlay and ratio panel.\"\"\"
    # Build histograms with these parameters
    signal_hist = build_hist_with_cuts(signal_jax, opt_params, m_ttbar_bins)
    ttbar_hist = build_hist_with_cuts(ttbar_jax, opt_params, m_ttbar_bins)
    wjets_hist = build_hist_with_cuts(wjets_jax, opt_params, m_ttbar_bins)
    data_hist = build_hist_with_cuts(data_jax, opt_params, m_ttbar_bins)

    # Get best-fit parameters from evermore
    # Build the statistical model
    model = evm.Model()

    # Define processes with normalization parameters
    mu = model.add_parameter("mu", value=1.0, lower=0.0)  # Signal strength
    kappa_ttbar = model.add_parameter("kappa_ttbar", value=1.0, lower=0.0)  # ttbar normalization

    # Background processes
    bkg_wjets = model.add_process("wjets", signal=False)
    bkg_wjets.samples["SR"] = evm.Sample.from_array(wjets_hist)

    bkg_ttbar = model.add_process("ttbar", signal=False)
    bkg_ttbar.samples["SR"] = evm.Sample.from_array(ttbar_hist)
    bkg_ttbar.set_modifier("normfactor", kappa_ttbar, domain="SR")

    # Signal process
    sig_proc = model.add_process("zprime", signal=True)
    sig_proc.samples["SR"] = evm.Sample.from_array(signal_hist)
    sig_proc.set_modifier("normfactor", mu, domain="SR")

    # Observed data
    model.set_data("SR", data_hist)

    # Fit: profile likelihood at μ=1 (assuming signal present)
    nll = model.negative_log_likelihood()

    # Find best-fit nuisance parameters (κ_ttbar) at μ=1
    import optimistix as optx

    def nll_nuisance(kappa):
        model.set_parameter("mu", 1.0)
        model.set_parameter("kappa_ttbar", kappa[0])
        return model.negative_log_likelihood()

    # Minimize over nuisance parameters
    solver = optx.BFGS(rtol=1e-4, atol=1e-4)
    initial_kappa = jnp.array([1.0])
    solution = optx.minimise(nll_nuisance, solver, initial_kappa, max_steps=50)
    best_kappa = float(solution.value[0])

    # Also get best-fit μ with profiled κ
    def nll_mu(mu_val):
        model.set_parameter("mu", mu_val[0])
        # Re-profile kappa for this mu
        sol = optx.minimise(
            lambda k: (model.set_parameter("kappa_ttbar", k[0]), model.negative_log_likelihood())[1],
            solver,
            jnp.array([best_kappa]),
            max_steps=50
        )
        return model.negative_log_likelihood()

    mu_solution = optx.minimise(nll_mu, solver, jnp.array([1.0]), max_steps=50)
    best_mu = float(mu_solution.value[0])

    # Set to best-fit values for plotting
    model.set_parameter("mu", best_mu)
    model.set_parameter("kappa_ttbar", best_kappa)

    # Get expected counts with best-fit parameters
    expected = model.expected_counts("SR")
    total_model = jnp.array(expected)

    # Separate components at best-fit
    wjets_fit = jnp.array(wjets_hist)  # W+jets has no modifier
    ttbar_fit = jnp.array(ttbar_hist) * best_kappa
    signal_fit = jnp.array(signal_hist) * best_mu

    bin_centers = (m_ttbar_bins[:-1] + m_ttbar_bins[1:]) / 2
    bin_width = m_ttbar_bins[1] - m_ttbar_bins[0]

    # Create main plot and ratio panel
    ax_main = fig.add_subplot(gs[0, col_idx])
    ax_ratio = fig.add_subplot(gs[1, col_idx], sharex=ax_main)

    # Main plot: Stacked backgrounds
    ax_main.stairs(wjets_fit, m_ttbar_bins, baseline=0, fill=True,
                   label='W+jets', color='#72A1E5', alpha=0.7, linewidth=0)
    ax_main.stairs(ttbar_fit, m_ttbar_bins, baseline=wjets_fit, fill=True,
                   label=r'$t\\bar{t}$', color='#907AD6', alpha=0.7, linewidth=0)

    # Signal (scaled for visibility)
    ax_main.stairs(signal_fit * 5, m_ttbar_bins, baseline=0,
                   label=f'Signal×5 (μ={best_mu:.2f})', color='red', linewidth=2, linestyle='--')

    # Total model
    ax_main.stairs(total_model, m_ttbar_bins, color='black',
                   linewidth=2, linestyle='-', label='S+B model', zorder=10)

    # Data points
    data_err = jnp.sqrt(data_hist)
    ax_main.errorbar(bin_centers, data_hist, yerr=data_err,
                     fmt='ko', label='Data', markersize=6, linewidth=1.5, zorder=15)

    ax_main.set_ylabel('Events / {:.0f} GeV'.format(bin_width), fontsize=13)
    ax_main.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax_main.legend(fontsize=10, ncol=2, loc='upper right')
    ax_main.grid(alpha=0.3, linestyle='--')
    ax_main.tick_params(labelbottom=False)

    # Add fit results text
    fit_text = (f"Best-fit:\\n"
                f"μ = {best_mu:.3f}\\n"
                f"κ_ttbar = {best_kappa:.3f}")
    ax_main.text(0.05, 0.97, fit_text, transform=ax_main.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Add parameter text
    param_text = (f"MET>{opt_params['met_cut']:.0f}\\n"
                  f"btag>{opt_params['btag_cut']:.2f}\\n"
                  f"HT>{opt_params['ht_cut']:.0f}\\n"
                  f"BW={opt_params['bandwidth']:.0f}")
    ax_main.text(0.95, 0.97, param_text, transform=ax_main.transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Ratio panel: Data / Model
    ratio = jnp.where(total_model > 0, data_hist / total_model, 1.0)
    ratio_err = jnp.where(total_model > 0, data_err / total_model, 0.0)

    ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_err,
                      fmt='ko', markersize=6, linewidth=1.5)
    ax_ratio.axhline(1, color='black', linestyle='-', linewidth=1.5)
    ax_ratio.axhline(1.1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_ratio.axhline(0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax_ratio.set_xlabel(r'$m_{t\\bar{t}}$ [GeV]', fontsize=13)
    ax_ratio.set_ylabel('Data / Model', fontsize=11)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(alpha=0.3, linestyle='--')
    ax_ratio.axhspan(0.9, 1.1, alpha=0.2, color='yellow', zorder=0)

    return best_mu, best_kappa

# Create figure with gridspec for ratio panels
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)

# Before optimization
console.print("  [cyan]Fitting before optimization...[/cyan]")
mu_before, kappa_before = plot_mttbar_with_fit(
    fig, gs, initial_params,
    f'Before Optimization (Z={Z_initial:.2f}σ)', 0
)

# After optimization
console.print("  [cyan]Fitting after optimization...[/cyan]")
mu_after, kappa_after = plot_mttbar_with_fit(
    fig, gs, params,
    f'After Optimization (Z={final_sig:.2f}σ)', 1
)

plt.show()

console.print("[green]✓ Before/after comparison complete![/green]")
console.print(f"  Significance: {Z_initial:.2f}σ → {final_sig:.2f}σ ({((final_sig - Z_initial)/Z_initial*100):.1f}% improvement)")
console.print(f"  Best-fit μ: {mu_before:.3f} → {mu_after:.3f}")
console.print(f"  Best-fit κ_ttbar: {kappa_before:.3f} → {kappa_after:.3f}")"""

        nb['cells'][i]['source'] = new_source
        print(f"✓ Updated cell {i} with ratio panel and best-fit parameters")
        break

# Save
with open('GRAEP_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("\n✓ Notebook updated with improved before/after plots!")
print("\nNew features:")
print("  • Ratio panel (Data/Model) with uncertainties")
print("  • Best-fit μ (signal strength) displayed")
print("  • Best-fit κ_ttbar (ttbar normalization) displayed")
print("  • Proper error bands in ratio")
print("  • Uses evermore's full fit capabilities")
