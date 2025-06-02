from typing import Dict, Any, Tuple, Optional, NamedTuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
import awkward as ak
import evermore as evm


class DifferentiablePhysicsSelections:
    """
    JAX-compatible physics selections that can be differentiated.
    Key: Convert ALL awkward arrays to pure JAX arrays before any JAX operations.
    """
    
    def __init__(self, config):
        self.config = config
    
    @partial(jax.jit, static_argnums=(0,))
    def soft_selection_cuts(self, params, jax_data):
        """
        Differentiable version of your soft cuts.
        
        Parameters:
        -----------
        params : dict
            Differentiable parameters (cut thresholds, etc.)
        jax_data : dict  
            Pure JAX arrays (no awkward arrays here!)
        """
        
        # All inputs are now pure JAX arrays
        region_met_pt = jax_data['met_pt']
        region_jets_btag = jax_data['jets_btag'] 
        region_lep_ht = jax_data['lep_ht']
        
        # Soft cuts with differentiable thresholds
        cuts = {
            'met_cut': jax.nn.sigmoid(
                (region_met_pt - params['met_threshold']) / 25.0
            ),
            'btag_cut': jax.nn.sigmoid(
                (region_jets_btag - params['btag_threshold']) * 10
            ),
        }
        
        # Combine cuts (product gives intersection-like behavior)
        cut_values = jnp.stack([cuts['met_cut'], cuts['btag_cut']])
        selection_weight = jnp.prod(cut_values, axis=0)
        
        return selection_weight, cuts

    @partial(jax.jit, static_argnums=(0,))
    def differentiable_observable(self, params, jax_data):
        """
        Compute observables in a differentiable way.
        """
        # Your existing observable calculation but JAX-compatible
        muon_pt = jax_data['muon_pt']
        jet_pt_sum = jax_data['jet_pt_sum']  # Pre-computed sum
        met_pt = jax_data['met_pt']
        
        # Example: parameterized observable calculation
        observable = (
            params['muon_weight'] * muon_pt + 
            0.1 * jet_pt_sum + 
            1. * met_pt
        )
        
        return observable

def convert_awkward_to_jax(obj_copies, mask):
    """
    Convert awkward arrays to pure JAX arrays BEFORE any JAX tracing.
    This is the crucial step that prevents the error.
    """
    # Apply mask first, then convert to numpy, then to JAX
    masked_objects = {k: v[mask] for k, v in obj_copies.items()}
    
    # Extract scalar quantities (one per event)
    met_pt = np.array(ak.to_numpy(masked_objects["PuppiMET"].pt))
    muon_pt = np.array(ak.to_numpy(ak.fill_none(ak.firsts(masked_objects["Muon"].pt), 0.0)))
    
    # Handle jagged arrays by taking sums/means
    jet_pt_sum = np.array(ak.to_numpy(ak.sum(masked_objects["Jet"].pt, axis=1)))
    jets_btag_mean = np.array(ak.to_numpy(ak.mean(masked_objects["Jet"].btagDeepB, axis=1)))
    
    # Compute derived quantities
    lep_ht = muon_pt + met_pt
    
    # Return pure JAX arrays
    jax_data = {
        'met_pt': jnp.array(met_pt),
        'muon_pt': jnp.array(muon_pt),
        'jet_pt_sum': jnp.array(jet_pt_sum),
        'jets_btag': jnp.array(jets_btag_mean),
        'lep_ht': jnp.array(lep_ht)
    }
    
    return jax_data


class DifferentiableZprimeAnalysis:
    """
    Extends your existing analysis with differentiable components.
    """
    
    def __init__(self, config):
        self.config = config
        self.diff_selections = DifferentiablePhysicsSelections(config)
        
        # Differentiable parameters (including process scales)
        self.params = {
            'met_threshold': 50.0,
            'btag_threshold': 0.5,
            'muon_weight': 1.0,
            'kde_bandwidth': 10.0,
            # Process-specific scales (cross-section * luminosity / n_events)
            'signal_scale': evm.Parameter(1.0),
            'ttbar_scale': evm.Parameter(1.0),  # Example using Evermore's Parameter
            # Systematic uncertainties
            'signal_systematic': 0.05,  # 5% on signal
            'background_systematic': 0.1,  # 10% on background
        }
    
    def differentiable_event_loop(self, params, process_data_dict):
        """
        The core differentiable part of your pipeline.
        Processes multiple processes (signal, backgrounds, data) and returns significance.
        
        Parameters:
        -----------
        params : dict
            Differentiable parameters (cut thresholds, etc.)
        process_data_dict : dict
            Dictionary with keys like 'signal', 'ttbar', 'wjets', 'data'
            Each containing JAX arrays for that process
        """
        
        histograms = {}
        
        # Process each sample separately
        for process_name, jax_data in process_data_dict.items():
            if len(jax_data['met_pt']) == 0:  # Skip empty processes
                continue
                
            # Apply differentiable selections
            selection_weight, cuts = self.diff_selections.soft_selection_cuts(
                params, jax_data
            )
            
            # Compute differentiable observable
            observable_vals = self.diff_selections.differentiable_observable(
                params, jax_data
            )
            
            # Differentiable binning
            # bins = jnp.linspace(0, 1000, 51)  # Example binning for your observable
            bins = jnp.linspace(0, 500, 26)  # Example binning for your observable
            bandwidth = params['kde_bandwidth']
            
            # KDE-style soft binning
            cdf = jax.scipy.stats.norm.cdf(
                bins.reshape(-1, 1),
                loc=observable_vals.reshape(1, -1),
                scale=bandwidth
            )
            
            # Weight each event's contribution by selection weight
            weighted_cdf = cdf * selection_weight.reshape(1, -1)
            bin_weights = weighted_cdf[1:, :] - weighted_cdf[:-1, :]
            histogram = jnp.sum(bin_weights, axis=1)
            
            # # Apply process-specific scaling
            # if process_name != 'data':
            #     # For MC: apply cross-section weights, luminosity, etc.
            #     process_scale = params.get(f'{process_name}_scale', 1.0)
            #     histogram = histogram * process_scale
            
            histograms[process_name] = histogram
        
        # Calculate significance from the histograms
        # significance = self._calculate_significance(params, histograms)
        significance = self._calculate_significance_evermore(params, histograms)
        
        return significance
    

    def _calculate_significance(self, params, histograms):
        """
        Calculate significance from process histograms.
        
        Parameters:
        -----------
        params : dict
            Parameters (including signal/background region definitions)
        histograms : dict
            Histograms for each process
        """
        
        # Define signal and background regions
        bins = jnp.linspace(0, 1000, 51)
        signal_region_mask = (bins[:-1] >= 400) & (bins[:-1] <= 600)
        background_region_mask = ((bins[:-1] >= 200) & (bins[:-1] <= 400)) | \
                                ((bins[:-1] >= 600) & (bins[:-1] <= 800))
        
        # Sum all background processes
        total_background = jnp.zeros_like(bins[:-1])
        signal_histogram = jnp.zeros_like(bins[:-1])
        
        for process_name, histogram in histograms.items():
            if process_name == 'data':
                continue  # Data used for validation, not in significance calc
            elif process_name in ['signal', 'zprime']:  # Signal processes
                signal_histogram = signal_histogram + histogram
            else:  # Background processes (ttbar, wjets, etc.)
                total_background = total_background + histogram
        
        # Calculate yields in signal and background regions
        signal_yield = jnp.sum(signal_histogram * signal_region_mask)
        background_yield = jnp.sum(total_background * signal_region_mask)
        
        # Add systematic uncertainties
        signal_syst = params.get('signal_systematic', 0.05) * signal_yield
        background_syst = params.get('background_systematic', 0.1) * background_yield
        
        # Significance calculation: S/√(B + δS² + δB²)
        denominator = jnp.sqrt(
            background_yield + signal_syst**2 + background_syst**2 + 1e-6
        )
        significance = signal_yield / denominator
        
        return significance


    def _calculate_significance_evermore(self, params, histograms):
        """
        Calculate significance using Evermore's statistical tools.
        
        Parameters:
        -----------
        params : dict
            Parameters (including signal/background region definitions)
        histograms : dict
            Histograms for each process
        """

        evm_params = {
            "signal_scale": params['signal_scale'],
            "ttbar_scale": params['ttbar_scale'],
        }

        def model(params: PyTree, hists: dict[str, Array]) -> Array:
            signal_modifier = params["signal_scale"].scale()
            ttbar_modifier = params["ttbar_scale"].scale()
            return (
                signal_modifier(hists['signal']) +
                ttbar_modifier(hists['ttbar']) +
                hists['wjets']# Assume wjets is not scaled
            )
            # return hists['signal'] + hists['ttbar'] + hists['wjets']

        # @jax.jit
        def loss(params: PyTree, hists: dict[str, Array], observation: Array) -> Array:
            expectation = model(params, hists)
            print(f"Expectation: {expectation}")
            print(f"Observation: {observation}")
            print(f"loss = {evm.pdf.Poisson(lamb=expectation).log_prob(observation).sum()}")
            return evm.pdf.Poisson(lamb=expectation).log_prob(observation).sum()
            # # Poisson NLL of the expectation and observation
            # log_likelihood = evm.pdf.Poisson(lamb=expectation).log_prob(observation).sum()
            # # Add parameter constraints from logpdfs
            # constraints = evm.loss.get_log_probs(params)
            # log_likelihood += evm.util.sum_over_leaves(constraints)
            # return -jnp.sum(log_likelihood)

        # observation = histograms.pop('data')
        observation = histograms["data"]
        histograms_ = {k: v for k, v in histograms.items() if k != 'data'}

        return loss(evm_params, histograms_, observation)

    
    def process_with_gradients(self, process_data_dict):
        """
        Modified version that processes multiple processes and calculates significance.
        
        Parameters:
        -----------
        process_data_dict : dict
            Dictionary with process names as keys and JAX data as values
            e.g., {'signal': jax_data, 'ttbar': jax_data, 'wjets': jax_data}
        """
        
        print("Running differentiable event loop for multiple processes...")
        print(f"Processes: {list(process_data_dict.keys())}")
        
        # Compute significance
        significance = self.differentiable_event_loop(self.params, process_data_dict)
        
        # Create gradient function
        grad_fn = jax.grad(self.differentiable_event_loop, argnums=0)
        gradients = grad_fn(self.params, process_data_dict)
        
        print(f"Signal significance: {significance}")
        print(f"Parameter gradients: {gradients}")
        
        return significance, gradients


def create_mock_multiprocess_data():
    """
    Create mock data for multiple processes (signal, backgrounds, data).
    """
    np.random.seed(42)
    
    processes = {
        'signal': {'n_events': 500, 'met_mean': 80, 'observable_mean': 500},
        'ttbar': {'n_events': 6000, 'met_mean': 60, 'observable_mean': 300},
        'wjets': {'n_events': 9000, 'met_mean': 45, 'observable_mean': 250},
        'data': {'n_events': 15000, 'met_mean': 55, 'observable_mean': 280}
    }
    
    process_data_dict = {}
    
    for process_name, config in processes.items():
        n_events = config['n_events']
        
        if n_events == 0:
            # Empty process
            process_data_dict[process_name] = {
                'met_pt': jnp.array([]),
                'muon_pt': jnp.array([]),
                'jet_pt_sum': jnp.array([]),
                'jets_btag': jnp.array([]),
                'lep_ht': jnp.array([])
            }
            continue
        
        # Generate realistic physics distributions for each process
        met_pt = np.random.exponential(config['met_mean'], n_events)
        muon_pt = np.random.exponential(60, n_events) + 20  # Minimum muon pT
        
        # Jets: different for each process
        jet_multiplicity = np.random.poisson(3, n_events) + 1
        jet_pt_sum = np.array([
            np.sum(np.random.exponential(40, mult)) 
            for mult in jet_multiplicity
        ])
        
        # B-tagging: signal has more b-jets
        if process_name == 'signal':
            jets_btag = np.random.beta(2, 1, n_events)  # More b-jets
        elif process_name == 'ttbar':
            jets_btag = np.random.beta(1.5, 2, n_events)  # Some b-jets
        else:
            jets_btag = np.random.beta(0.5, 3, n_events)  # Fewer b-jets
        
        lep_ht = muon_pt + met_pt
        
        # Apply baseline selection (realistic acceptance)
        baseline_mask = (
            (met_pt > 30) & 
            (muon_pt > 55) & 
            (jet_pt_sum > 100) &
            (np.random.random(n_events) > 0.1)  # Additional inefficiencies
        )
        
        # Convert to JAX arrays
        process_data_dict[process_name] = {
            'met_pt': jnp.array(met_pt[baseline_mask]),
            'muon_pt': jnp.array(muon_pt[baseline_mask]),
            'jet_pt_sum': jnp.array(jet_pt_sum[baseline_mask]),
            'jets_btag': jnp.array(jets_btag[baseline_mask]),
            'lep_ht': jnp.array(lep_ht[baseline_mask])
        }
        
        print(f"{process_name}: {n_events} → {len(process_data_dict[process_name]['met_pt'])} events after selection")
    
    return process_data_dict


def optimize_analysis_cuts():
    """
    Example of how to optimize analysis cuts using multiple processes to maximize significance.
    """
    
    analysis = DifferentiableZprimeAnalysis(config={})
    
    process_data_dict = create_mock_multiprocess_data()
    
    print("\nTesting differentiable processing...")
    significance, gradients = analysis.process_with_gradients(process_data_dict)
    
    print(f"Initial significance: {significance:.4f}")
    print(f"Initial gradients: {gradients}")
    
    # The objective is the differentiable_event_loop itself
    def objective(params):
        return analysis.differentiable_event_loop(params, process_data_dict)
    
    print("\nRunning optimization to maximize significance...")
    
    # Simple gradient ascent with parameter constraints
    learning_rate = 0.001
    params = analysis.params.copy()
    
    print(f"{'Step':>4} {'Significance':>12} {'MET Cut':>8} {'B-tag Cut':>10} {'Lep HT Cut':>11}")
    print("-" * 55)
    
    for i in range(25):
    # for i in range(1):
        significance = objective(params)
        grads = jax.grad(objective)(params)
        
        # Update parameters with constraints
        for key, param in params.items():
            # check if param is an evermore.Parameter
            print(f"Processing parameter {key} with value {param} and gradient {grads[key]}")
            if isinstance(param, evm.Parameter):
                params[key] = evm.Parameter(
                    value=param.value + learning_rate * grads[key].value,
                    lower=param.lower,
                    upper=param.upper
                )

            elif key.endswith('_threshold'):
                # For cut thresholds, use smaller learning rate and constrain ranges
                if key == 'met_threshold':
                    params[key] = jnp.clip(
                        params[key] + learning_rate * grads[key],
                        20.0, 150.0
                    )
                elif key == 'btag_threshold':
                    params[key] = jnp.clip(
                        params[key] + learning_rate * grads[key],
                        0.1, 0.9
                    )
            elif key.endswith('_weight'):
                # For weights, constrain to positive values
                params[key] = jnp.maximum(
                    params[key] + learning_rate * grads[key],
                    0.01
                )
            elif key.endswith('_scale'):
                # Process scales should stay positive and reasonable
                params[key] = jnp.clip(
                    params[key] + learning_rate * grads[key],
                    0.1, 10.0
                )
            else:
                # Other parameters
                params[key] = params[key] + learning_rate * grads[key]
        
        # if (i + 1) % 5 == 0 or i == 0:
        if True:
            print(f"{i+1:4d} {significance:12.4f} {params['met_threshold']:8.1f} "
                  f"{params['btag_threshold']:10.3f}")
    
    final_significance = objective(params)
    print(f"\nOptimization complete!")
    print(f"Initial significance: {significance:.4f}")
    print(f"Final significance: {final_significance:.4f}")
    print(f"Improvement: {((final_significance/significance - 1) * 100):.1f}%")
    
    print(f"\nOptimized parameters:")
    for key, value in params.items():
        if isinstance(value, (int, float)) or hasattr(value, 'item'):
            print(f"  {key}: {float(value):.4f}")
    
    # Show process contributions at optimal cuts
    print(f"\nProcess contributions at optimal cuts:")
    histograms = {}
    for process_name, jax_data in process_data_dict.items():
        if len(jax_data['met_pt']) == 0:
            continue
        selection_weight, _ = analysis.diff_selections.soft_selection_cuts(params, jax_data)
        total_weight = jnp.sum(selection_weight)
        print(f"  {process_name}: {float(total_weight):.1f} events")
    
    return params, final_significance


if __name__ == "__main__":
    optimize_analysis_cuts()