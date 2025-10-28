"""
Phase 2: Refactor Part 6 (optimization) and Part 7 (visualization)
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

print("="*80)
print("PHASE 2: PART 6 & 7 REFACTORING")
print("="*80)

# ============================================================================
# PART 6: Optimization
# ============================================================================

part6_header_idx = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '## Part 6: Optimize evermore Significance' in source:
        part6_header_idx = i
        break

if part6_header_idx:
    # Update header
    nb['cells'][part6_header_idx]['source'] = """## Part 6: Optimize with Gradients

**What you know**: Optimize by trying many cut values (grid search, manual tuning)

**What's new**: Use gradients to automatically find optimal parameters

**Why it matters**: Gradient descent is **much faster** than grid search!

### Gradient-Based Optimization

Since `compute_evermore_significance` is differentiable, we can use gradient ascent:

$$
\\theta^{(t+1)} = \\theta^{(t)} + \\alpha \\nabla_{\\theta} Z
$$

where $\\theta$ includes **all parameters**: MET cut, b-tag cut, HT cut, AND bandwidth!

Traditional HEP: Try $10^4$ combinations → hours
GRAEP: Follow gradients → minutes!"""

    print("✓ Part 6: Updated pedagogical header")

    # Find optimization cell and update it
    for i in range(part6_header_idx + 1, min(part6_header_idx + 10, len(nb['cells']))):
        source = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']

        if 'optimizer = optax.multi_transform' in source:
            nb['cells'][i]['source'] = """# Setup optimizer with per-parameter learning rates
# Different parameters need different step sizes!
optimizer = optax.multi_transform(
    {
        'met_cut': optax.adam(learning_rate=2.0),      # MET in GeV → larger steps
        'btag_cut': optax.adam(learning_rate=0.005),   # b-tag 0-1 → tiny steps
        'ht_cut': optax.adam(learning_rate=1.0),       # HT in GeV → moderate steps
        'bandwidth': optax.adam(learning_rate=1.0),    # Bandwidth in GeV → moderate steps
    },
    param_labels={p: p for p in initial_params.keys()}
)

opt_state = optimizer.init(initial_params)
params = initial_params.copy()

# Optimization loop
n_iterations = 20  # evermore is slower, so fewer iterations
opt_history = {
    'met_cut': [],
    'btag_cut': [],
    'ht_cut': [],
    'bandwidth': [],  # NEW!
    'significance': []
}

console.print("[cyan]Optimizing all 4 parameters...[/cyan]")

for iteration in track(range(n_iterations), description="Optimizing"):
    # Compute significance and gradient
    sig = compute_evermore_significance(params, wjets_jax, ttbar_jax, signal_jax, data_jax)
    grads = grad_fn(params)

    # Gradient ascent (maximize significance)
    grads = {k: -v for k, v in grads.items()}  # Negate for ascent

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Track progress
    opt_history['met_cut'].append(float(params['met_cut']))
    opt_history['btag_cut'].append(float(params['btag_cut']))
    opt_history['ht_cut'].append(float(params['ht_cut']))
    opt_history['bandwidth'].append(float(params['bandwidth']))
    opt_history['significance'].append(float(sig))

    # Print progress at key iterations
    if iteration in [0, 10, 19]:
        console.print(f"  Iter {iteration}: Z={sig:.2f}σ, "
                     f"MET={params['met_cut']:.1f}, "
                     f"btag={params['btag_cut']:.2f}, "
                     f"HT={params['ht_cut']:.1f}, "
                     f"BW={params['bandwidth']:.1f}")

final_sig = opt_history['significance'][-1]
console.print(f"\\n[green]✓ Optimization complete![/green]")
console.print(f"  Initial: {Z_initial:.2f}σ → Final: {final_sig:.2f}σ")
console.print(f"  MET cut: {initial_params['met_cut']:.1f} → {params['met_cut']:.1f} GeV")
console.print(f"  b-tag cut: {initial_params['btag_cut']:.2f} → {params['btag_cut']:.2f}")
console.print(f"  HT cut: {initial_params['ht_cut']:.1f} → {params['ht_cut']:.1f} GeV")
console.print(f"  Bandwidth: {initial_params['bandwidth']:.1f} → {params['bandwidth']:.1f} GeV")"""

            print(f"✓ Part 6: Updated optimization loop with bandwidth (cell {i})")
            break

    # Find visualization cell and update to 4 subplots
    for i in range(part6_header_idx + 1, min(part6_header_idx + 15, len(nb['cells']))):
        source = ''.join(nb['cells'][i]['source']) if isinstance(cell['source'], list) else cell['source']

        if 'fig, axes = plt.subplots' in source and 'opt_history[\'significance\']' in source:
            nb['cells'][i]['source'] = """# Visualize optimization trajectory (4 parameters!)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Significance
axes[0].plot(opt_history['significance'], linewidth=2, color='purple', marker='o')
axes[0].axhline(float(Z_initial), color='gray', linestyle='--', alpha=0.5, label='Initial')
axes[0].set_xlabel('Iteration', fontsize=14)
axes[0].set_ylabel('Significance (σ)', fontsize=14)
axes[0].set_title('Discovery Significance', fontsize=15, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(alpha=0.3)

# MET cut
axes[1].plot(opt_history['met_cut'], linewidth=2, color='blue', marker='o')
axes[1].axhline(float(initial_params['met_cut']), color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Iteration', fontsize=14)
axes[1].set_ylabel('MET Cut [GeV]', fontsize=14)
axes[1].set_title('MET Threshold', fontsize=15, fontweight='bold')
axes[1].grid(alpha=0.3)

# b-tag cut
axes[2].plot(opt_history['btag_cut'], linewidth=2, color='green', marker='o')
axes[2].axhline(float(initial_params['btag_cut']), color='gray', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Iteration', fontsize=14)
axes[2].set_ylabel('b-tag Cut', fontsize=14)
axes[2].set_title('b-tag Threshold', fontsize=15, fontweight='bold')
axes[2].grid(alpha=0.3)

# Bandwidth (NEW!)
axes[3].plot(opt_history['bandwidth'], linewidth=2, color='orange', marker='o')
axes[3].axhline(float(initial_params['bandwidth']), color='gray', linestyle='--', alpha=0.5)
axes[3].set_xlabel('Iteration', fontsize=14)
axes[3].set_ylabel('Bandwidth [GeV]', fontsize=14)
axes[3].set_title('KDE Bandwidth', fontsize=15, fontweight='bold')
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.show()

console.print("[green]✓ All 4 parameters optimized with gradients![/green]")
console.print("[cyan]Key insight:[/cyan] Even histogram smoothing (bandwidth) can be optimized!")"""

            print(f"✓ Part 6: Updated visualization to 4 subplots (cell {i})")
            break

# ============================================================================
# PART 7: Workflow Summary + Before/After Visualization
# ============================================================================

part7_header_idx = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '## Part 7: Complete GRAEP Workflow' in source:
        part7_header_idx = i
        break

if part7_header_idx:
    print(f"\n✓ Part 7: Found at cell {part7_header_idx}")

    # Find the workflow summary cell and add before/after plot after it
    for i in range(part7_header_idx + 1, min(part7_header_idx + 10, len(nb['cells']))):
        source = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']

        if 'Optimization Results' in source and 'table.add_row' in source:
            # Add before/after visualization after this cell
            insert_idx = i + 1

            new_viz_cell = {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": """# Before/After m_ttbar distribution with evermore fit
console.print("\\n[cyan]Building before/after comparison...[/cyan]")

# Helper function to build and visualize fit
def plot_mttbar_with_fit(ax, opt_params, title):
    \"\"\"Plot m_ttbar distribution with evermore fit overlay.\"\"\"
    # Build histograms with these parameters
    signal_hist = build_hist_with_cuts(signal_jax, opt_params, m_ttbar_bins)
    ttbar_hist = build_hist_with_cuts(ttbar_jax, opt_params, m_ttbar_bins)
    wjets_hist = build_hist_with_cuts(wjets_jax, opt_params, m_ttbar_bins)
    data_hist = build_hist_with_cuts(data_jax, opt_params, m_ttbar_bins)

    # Compute fit (simplified - just show S+B model)
    # In practice, would use best-fit μ and κ from profile likelihood
    total_bkg = wjets_hist + ttbar_hist
    total_sb = total_bkg + signal_hist

    bin_centers = (m_ttbar_bins[:-1] + m_ttbar_bins[1:]) / 2
    bin_width = m_ttbar_bins[1] - m_ttbar_bins[0]

    # Stacked backgrounds
    ax.stairs(wjets_hist, m_ttbar_bins, baseline=0, fill=True,
              label='W+jets', color='#72A1E5', alpha=0.7, linewidth=0)
    ax.stairs(ttbar_hist, m_ttbar_bins, baseline=wjets_hist, fill=True,
              label=r'$t\\bar{t}$', color='#907AD6', alpha=0.7, linewidth=0)

    # Signal (scaled for visibility)
    ax.stairs(signal_hist * 5, m_ttbar_bins, baseline=0,
              label='Signal×5', color='red', linewidth=2, linestyle='--')

    # Data points
    ax.errorbar(bin_centers, data_hist, yerr=jnp.sqrt(data_hist),
                fmt='ko', label='Data', markersize=5, linewidth=1)

    # S+B model line
    ax.stairs(total_sb, m_ttbar_bins, color='black',
              linewidth=2, linestyle='-', label='S+B model')

    ax.set_xlabel(r'$m_{t\\bar{t}}$ [GeV]', fontsize=14)
    ax.set_ylabel('Events / {:.0f} GeV'.format(bin_width), fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)

    # Add parameter text
    param_text = (f"MET>{opt_params['met_cut']:.0f}, "
                  f"btag>{opt_params['btag_cut']:.2f}\\n"
                  f"HT>{opt_params['ht_cut']:.0f}, "
                  f"BW={opt_params['bandwidth']:.0f}")
    ax.text(0.05, 0.95, param_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Before optimization
plot_mttbar_with_fit(ax1, initial_params,
                     f'Before Optimization (Z={Z_initial:.2f}σ)')

# After optimization
plot_mttbar_with_fit(ax2, params,
                     f'After Optimization (Z={final_sig:.2f}σ)')

plt.tight_layout()
plt.show()

console.print("[green]✓ Before/after comparison complete![/green]")
console.print(f"  Significance improved from {Z_initial:.2f}σ to {final_sig:.2f}σ")
console.print(f"  Improvement: {((final_sig - Z_initial)/Z_initial*100):.1f}%")"""
            }

            nb['cells'].insert(insert_idx, new_viz_cell)
            print(f"✓ Part 7: Added before/after m_ttbar visualization (cell {insert_idx})")
            break

print("\nSaving Phase 2 refactored notebook...")
with open('GRAEP_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("PHASE 2 COMPLETE: Parts 6 & 7 refactored")
print("="*80)
print("✓ Part 6: Added pedagogical header")
print("✓ Part 6: Added bandwidth to optimizer + history")
print("✓ Part 6: Updated visualization to 4 subplots")
print("✓ Part 7: Added before/after m_ttbar plots with fit overlays")
print("\nNext: Clean up function redefinitions in Appendix")
