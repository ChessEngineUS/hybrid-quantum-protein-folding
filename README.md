# Hybrid Quantum-AI Energy Fusion for Protein Structure Prediction

[![CI](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/actions/workflows/ci.yml/badge.svg)](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/actions/workflows/ci.yml)
[![Release](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/actions/workflows/release.yml/badge.svg)](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/ChessEngineUS/hybrid-quantum-protein-folding/branch/main/graph/badge.svg)](https://codecov.io/gh/ChessEngineUS/hybrid-quantum-protein-folding)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/qiskit-1.0%2B-purple)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-red)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-available-blue)](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/pkgs/container/hybrid-quantum-protein-folding)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChessEngineUS/hybrid-quantum-protein-folding/blob/main/notebooks/colab_quickstart.ipynb)

A novel hybrid framework integrating the Variational Quantum Eigensolver (VQE) with deep neural networks to address protein structure prediction in regimes where current classical methods fail: intrinsically disordered regions (IDRs), metamorphic proteins, and proteins under mechanical force.

## 🚀 Try It Now!

**Google Colab Notebooks** (no installation required):

- [![Open Quickstart](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChessEngineUS/hybrid-quantum-protein-folding/blob/main/notebooks/colab_quickstart.ipynb) **Quickstart** - Get started in 5 minutes
- [![Open Tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChessEngineUS/hybrid-quantum-protein-folding/blob/main/notebooks/tutorial.ipynb) **Full Tutorial** - Complete walkthrough

**Docker**:
```bash
docker pull ghcr.io/chessengineUS/hybrid-quantum-protein-folding:latest
docker run -it ghcr.io/chessengineUS/hybrid-quantum-protein-folding:latest
```

## 📖 Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Perfect for beginners! Step-by-step installation and first prediction
- **[Quick Start Guide](docs/QUICKSTART.md)** - Fast-track to using the framework with examples
- **[Architecture Documentation](docs/ARCHITECTURE.md)** - Detailed system design and implementation
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to the project
- **[Changelog](CHANGELOG.md)** - Version history and release notes

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

## 💻 Quick Start

### Installation

```bash
git clone https://github.com/ChessEngineUS/hybrid-quantum-protein-folding.git
cd hybrid-quantum-protein-folding
pip install -e .
```

**First time?** Check out the [Getting Started Guide](docs/GETTING_STARTED.md) for detailed setup instructions!

### Basic Usage

```python
from hqpf.models import HybridModel
import torch

# Define a protein sequence
sequence_str = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
             'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
             'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
sequence = torch.tensor([aa_to_idx[aa] for aa in sequence_str], dtype=torch.long)

# Initialize hybrid model
model = HybridModel(
    n_residues=len(sequence),
    n_qubits=20,
    use_quantum=False,  # Use simulator for quick testing
    device='cpu'
)

# Predict structure
model.eval()
with torch.no_grad():
    outputs = model(sequence, n_candidates=10)

best_structure = outputs['best_structure']
print(f"Predicted structure shape: {best_structure.shape}")
print(f"Best energy: {outputs['best_energy'].item():.4f}")
```

**More examples:** See [Quick Start Guide](docs/QUICKSTART.md) and [examples/](examples/) directory.

### Running Benchmarks

```bash
# Run on all target problems
python examples/benchmark_idrs.py --output_dir ./results

# Training from scratch
python examples/train_basic.py --config config/default_config.yaml

# Inference from checkpoint
python examples/inference_demo.py --sequence ACDEFGHIKL --checkpoint model.pt
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
- **Minimum**: 8GB RAM, CPU, 10GB storage
- **Recommended**: 32GB RAM, NVIDIA GPU (V100/A100), 100GB storage

## 📁 Repository Structure

```
hybrid-quantum-protein-folding/
├── hqpf/                            # Main package
│   ├── models/                      # Hybrid Q+classical models
│   ├── data/                        # Data loading and encoding
│   ├── training/                    # Training loops and losses
│   ├── quantum/                     # Quantum circuits and backends
│   ├── classical/                   # Classical energy functions
│   └── utils/                       # Metrics and visualization
├── examples/                        # Example scripts
│   ├── train_basic.py
│   ├── inference_demo.py
│   └── benchmark_idrs.py
├── notebooks/                       # Jupyter tutorials
│   ├── colab_quickstart.ipynb       # Google Colab quickstart
│   └── tutorial.ipynb               # Full tutorial
├── docs/                            # Documentation
│   ├── GETTING_STARTED.md
│   ├── QUICKSTART.md
│   └── ARCHITECTURE.md
├── .github/workflows/               # CI/CD pipelines
│   ├── ci.yml                       # Continuous integration
│   └── release.yml                  # Release automation
├── config/                          # Configuration files
│   └── default_config.yaml
├── tests/                           # Unit tests
├── Dockerfile                       # Docker container
├── requirements.txt                 # Dependencies
├── setup.py                         # Package setup
├── CONTRIBUTING.md                  # Contribution guidelines
├── CHANGELOG.md                     # Version history
├── LICENSE                          # MIT License
└── README.md                        # This file
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

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- How to report bugs
- How to suggest enhancements
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## 🎓 Learning Resources

### Tutorials
- [Getting Started Guide](docs/GETTING_STARTED.md) - Complete beginner's guide
- [Colab Quickstart](https://colab.research.google.com/github/ChessEngineUS/hybrid-quantum-protein-folding/blob/main/notebooks/colab_quickstart.ipynb) - Try it now!
- [Quick Start](docs/QUICKSTART.md) - Fast-track usage examples

### Background Reading
- **Quantum Computing**: [Qiskit Textbook](https://qiskit.org/textbook)
- **Protein Structure**: [PDB-101](https://pdb101.rcsb.org/)
- **VQE Algorithms**: [PennyLane VQE Tutorial](https://pennylane.ai/qml/demos/tutorial_vqe.html)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM Quantum** for hardware access and support
- **Baker Lab (IPD)** for protein design insights and methodologies
- **Aspuru-Guzik Group** for quantum chemistry frameworks
- **Open-source community** for PyTorch, Qiskit, and PennyLane

## 📧 Contact

**Tommaso R. Marena**  
GitHub: [@ChessEngineUS](https://github.com/ChessEngineUS)  
Project Link: [https://github.com/ChessEngineUS/hybrid-quantum-protein-folding](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding)

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/discussions)
- **Examples**: [examples/](examples/)
- **Notebooks**: [notebooks/](notebooks/)
- **CI/CD**: [Actions](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/actions)
- **Docker Images**: [Packages](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/pkgs/container/hybrid-quantum-protein-folding)

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

## 🗺️ Roadmap

- [x] Core framework implementation
- [x] VQE with learned Hamiltonians
- [x] Surrogate model for efficiency
- [x] Benchmark datasets
- [x] Example scripts
- [x] Comprehensive documentation
- [x] CI/CD pipelines
- [x] Docker containers
- [x] Google Colab notebooks
- [ ] Pre-trained models
- [ ] Web interface for predictions
- [ ] Integration with AlphaFold3
- [ ] Support for post-translational modifications
- [ ] Real-time quantum circuit monitoring
- [ ] Automated hyperparameter tuning

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

<div align="center">

**"Combining quantum superposition with learned priors to explore protein conformational space where classical methods fail."**

🧬 + ⚛️ + 🤖 = 🚀

[![CI Status](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/actions/workflows/ci.yml/badge.svg)](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChessEngineUS/hybrid-quantum-protein-folding/blob/main/notebooks/colab_quickstart.ipynb)

</div>
