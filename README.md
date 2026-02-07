# Hybrid Quantum-AI Energy Fusion for Protein Structure Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Qiskit](https://img.shields.io/badge/qiskit-1.0%2B-purple)
![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-red)

A novel hybrid framework integrating the Variational Quantum Eigensolver (VQE) with deep neural networks to address protein structure prediction in regimes where current classical methods fail: intrinsically disordered regions (IDRs), metamorphic proteins, and proteins under mechanical force.

## 🎯 Key Innovation

**Conditional, sequence-aware quantum Hamiltonians** trained end-to-end such that quantum-evaluated energies rank biologically relevant conformations above decoys, specifically in systems where AlphaFold-class predictors underperform.

Unlike prior work that applies pre-specified QUBO/lattice models to toy peptides, we **learn** the quantum Hamiltonian parameters jointly with a structure encoder, making quantum energy evaluation conditional on local sequence context and environmental state.

## 🔬 Target Problems

### Problem Set A: Intrinsically Disordered Regions
- **p53 Transactivation Domain (p53-TAD)**: ~60 aa, highly disordered, forms nascent helical structure upon interaction
- **Human c-Fos Disordered Linker**: ~40 aa, transient alpha-helical propensity, critical for kinetics

### Problem Set B: Metamorphic/Multi-State Proteins
- **Protein FF Domain**: Adopts distinct folds with different binding partners
- **p53 DNA-Binding Domain Mutants**: ΔΔG prediction for disease mutations

### Problem Set C: Proteins Under Constraint
- **Zika Virus Protein Fragment**: ~15 aa under mechanical force or crowding

## 🏗️ Architecture

### Module 1: Coarse-Grained Representation
- Lattice model (3.8 Å spacing) or hybrid off-lattice backbone dihedral encoding
- Fragment-based representation for N > 20 residues
- Self-avoiding walk constraints

### Module 2: Conditional Quantum Hamiltonian & VQE
- **Parametric Hamiltonian**:
  ```
  H_eff(θ) = H_local(θ) + H_contact(θ) + H_env(θ)
  ```
- **Local terms**: Sequence-dependent dihedral preferences
- **Non-local contact terms**: Learned hydrophobic, charge, distance interactions
- **Environment terms**: pH, ionic strength, mechanical force
- Hardware-efficient ansatz for NISQ devices (IBM Falcon/Heron, IonQ Harmony)

### Module 3: Energy Fusion with Structure Generator
- Graph Neural Network (GNN) structure generator
- Hybrid energy score:
  ```
  S(x) = E_classical(x) + λ·E_quantum(x; θ*) + L_prior(x|s)
  ```
- CNN surrogate model (90-95% reduction in quantum queries)
- End-to-end training via automatic differentiation

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ChessEngineUS/hybrid-quantum-protein-folding.git
cd hybrid-quantum-protein-folding
pip install -e .
```

### Basic Usage

```python
from hqpf.models import HybridQuantumModel
from hqpf.data import load_benchmark_protein
from hqpf.training import Trainer

# Load a benchmark protein (e.g., p53-TAD core region)
protein = load_benchmark_protein('p53_tad', residues=20)

# Initialize hybrid model
model = HybridQuantumModel(
    n_qubits=24,
    ansatz_depth=4,
    gnn_hidden_dim=128,
    use_surrogate=True
)

# Train on IBM quantum hardware
trainer = Trainer(
    model=model,
    backend='ibm_falcon',
    surrogate_refresh_interval=100
)

trainer.train(
    protein=protein,
    epochs=50,
    batch_size=8
)

# Predict structure
predicted_structure = model.predict(protein.sequence)
```

### Running Benchmarks

```bash
# Run on all target problems
python scripts/run_benchmark.py --problem_set all --backend ibm_falcon

# Specific problem with comparison to baselines
python scripts/run_benchmark.py --problem p53_tad --compare alphafold rosetta
```

## 📊 Performance Metrics

- **RMSD to native** (or NMR ensemble centroid)
- **ΔΔG ranking accuracy** for mutants
- **Ensemble diversity** for IDRs
- **Quantum circuit efficiency**: depth, gate count, QPU calls vs. accuracy

## 🔧 Hardware Requirements

### Quantum Backends
- **IBM Quantum**: Falcon (27 qubits), Heron (133 qubits) via IBM Quantum Cloud
- **IonQ**: 11-88 qubits via Azure Quantum or direct partnership
- **Fallback**: PennyLane simulators for development

### Classical Computing
- GPU: NVIDIA V100/A100 recommended for GNN training
- RAM: 32GB minimum
- Storage: 100GB for benchmark datasets

## 📁 Repository Structure

```
hybrid-quantum-protein-folding/
├── hqpf/
│   ├── __init__.py
│   ├── models/
│   │   ├── hybrid_model.py          # Main hybrid Q+classical model
│   │   ├── quantum_hamiltonian.py   # Parametric Hamiltonian
│   │   ├── vqe_solver.py            # VQE implementation
│   │   ├── structure_generator.py   # GNN structure decoder
│   │   └── surrogate.py             # CNN surrogate model
│   ├── data/
│   │   ├── coarse_grained.py        # Lattice/dihedral encoding
│   │   ├── benchmarks.py            # Target protein datasets
│   │   └── loaders.py               # Data loading utilities
│   ├── training/
│   │   ├── trainer.py               # Training loop
│   │   ├── losses.py                # Hybrid loss functions
│   │   └── optimizers.py            # Custom optimizers
│   ├── quantum/
│   │   ├── circuits.py              # Quantum circuit construction
│   │   ├── ansatz.py                # Hardware-efficient & PIA ansatz
│   │   ├── backends.py              # Backend management
│   │   └── error_mitigation.py      # ZNE, readout correction
│   ├── classical/
│   │   ├── energy_terms.py          # Classical potentials
│   │   └── baseline_models.py       # Rosetta, AlphaFold wrappers
│   └── utils/
│       ├── metrics.py               # RMSD, TM-score, ΔΔG
│       ├── visualization.py         # Structure plotting
│       └── logging.py               # Experiment tracking
├── scripts/
│   ├── run_benchmark.py
│   ├── train_hamiltonian.py
│   └── analyze_results.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_quantum_circuits.ipynb
│   └── 03_training_visualization.ipynb
├── tests/
│   ├── test_quantum_hamiltonian.py
│   ├── test_vqe.py
│   └── test_structure_generator.py
├── data/
│   └── benchmarks/                  # Benchmark protein data
├── configs/
│   └── default_config.yaml
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🧪 Experimental Validation

### Benchmark Datasets
- **PDB Ensemble**: NMR ensembles from PDB
- **BioMagResBank**: NMR HSQC, PRE data
- **Thermal Stability**: Tm, ΔΔG from literature
- **Single-Molecule**: FRET distance distributions

### Comparison Baselines
- AlphaFold2
- Rosetta
- Classical coarse-grained MD
- Quantum-only (unlearned Hamiltonian)

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{hybrid_quantum_protein_2026,
  title={Hybrid Quantum-AI Energy Fusion for Protein Structure Prediction: A Near-Term Quantum Advantage in Disorder and Metamorphic Regimes},
  author={Marena, Tommaso R.},
  year={2026},
  journal={arXiv preprint},
  note={Implementation available at https://github.com/ChessEngineUS/hybrid-quantum-protein-folding}
}
```

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IBM Quantum for hardware access
- Baker Lab for protein design insights
- Aspuru-Guzik group for quantum chemistry methodologies

## 📧 Contact

**Tommaso R. Marena**  
Email: [Your Email]  
GitHub: [@ChessEngineUS](https://github.com/ChessEngineUS)  
LinkedIn: [Your LinkedIn]

---

*"Combining quantum superposition with learned priors to explore protein conformational space where classical methods fail."*