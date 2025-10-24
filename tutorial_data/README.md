# GRAEP Tutorial Data: Searching for Z' → ttbar

## Overview

This dataset contains pre-processed events from CMS Open Data (2016) for teaching differentiable analysis methods in High Energy Physics.

## Physics Context

We're searching for a hypothetical **Z' boson** (a heavy cousin of the Z boson) that decays into a pair of top quarks. This would be evidence for physics beyond the Standard Model!

## Dataset Contents

- **wjets.npz**: 5,000 W+jets background events
- **ttbar.npz**: 10,000 ttbar background events
- **signal.npz**: 10,000 Z' signal events (various masses)

Total size: ~750 KB

## Features (12 per event)

Each event has these physics features computed from detector measurements:

| Feature | Description | Units |
|---------|-------------|-------|
| `n_jet` | Number of jets | count |
| `leading_jet_mass` | Mass of highest-pT jet | GeV |
| `subleading_jet_mass` | Mass of 2nd highest-pT jet | GeV |
| `leading_jet_pt` | Transverse momentum of leading jet | GeV |
| `subleading_jet_pt` | Transverse momentum of 2nd jet | GeV |
| `st` | Scalar sum HT (total pT) | GeV |
| `leading_jet_btag` | b-tagging discriminator | 0-1 |
| `subleading_jet_btag` | b-tag of 2nd jet | 0-1 |
| `met_pt` | Missing transverse energy | GeV |
| `muon_pt` | Muon transverse momentum | GeV |
| `S_zz` | Sphericity tensor component | dimensionless |
| `min_deltaR_mu_jet` | Min angular separation μ-jet | radians |

## Usage

```python
import numpy as np
from pathlib import Path

# Load data
data_dir = Path("/kaggle/input/graep-hep-tutorial-data")
wjets = np.load(data_dir / "wjets.npz")
ttbar = np.load(data_dir / "ttbar.npz")
signal = np.load(data_dir / "signal.npz")

# Access features
met_values = wjets['met_pt']
btag_scores = signal['leading_jet_btag']
```

## Learning Objectives

1. Train neural networks in JAX for event classification
2. Implement differentiable selection cuts
3. Optimize analysis using automatic differentiation
4. Maximize statistical significance (S/√B)

## Tutorial

See the accompanying notebook: **GRAEP Tutorial: Differentiable Analysis in HEP**

## Citation

If you use this data in research or education:

```
IRIS-HEP GRAEP Project (2024)
Differentiable Analysis for High Energy Physics
https://github.com/iris-hep/GRAEP
```

## Source

Original data from CMS Open Data Portal:
- Dataset: /SingleMuon/Run2016X/ (Real Data)
- MC Simulation: NanoAOD format

## License

CC0: Public Domain
