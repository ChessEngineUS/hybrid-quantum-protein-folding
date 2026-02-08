# 🧬 Hybrid Quantum-AI Protein Folding

[![CI/CD](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/actions/workflows/comprehensive-ci.yml/badge.svg)](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/actions)
[![codecov](https://codecov.io/gh/ChessEngineUS/hybrid-quantum-protein-folding/branch/main/graph/badge.svg)](https://codecov.io/gh/ChessEngineUS/hybrid-quantum-protein-folding)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A hybrid quantum-classical deep learning framework for protein structure prediction, combining Variational Quantum Eigensolvers (VQE) with Graph Neural Networks (GNN).

## 🌟 Key Features

- ⚛️ **Quantum Energy Evaluation**: VQE-based conformational energy estimation
- 🕷️ **Neural Structure Generation**: GNN-powered 3D coordinate prediction
- ⚡ **End-to-End Differentiable**: Gradients flow through both quantum and classical components
- 🚀 **Production-Ready**: Comprehensive tests, CI/CD, and Docker support

## 🚀 Quick Start

### Google Colab (Fastest)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChessEngineUS/hybrid-quantum-protein-folding/blob/main/notebooks/colab_quickstart.ipynb)

### Local Installation

```bash
# Clone repository
git clone https://github.com/ChessEngineUS/hybrid-quantum-protein-folding.git
cd hybrid-quantum-protein-folding

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/
```

### Docker

```bash
docker build -t hqpf:latest .
docker run --gpus all -p 8888:8888 hqpf:latest
```

## 💻 Usage

### Basic Prediction

```python
import torch
from hqpf.models import HybridModel
from hqpf.data.benchmarks import load_benchmark_protein, AA_TO_IDX

# Load a benchmark protein
protein = load_benchmark_protein('peptide_10', dataset='test')
sequence = protein.to_tensor(AA_TO_IDX, device='cuda')

# Initialize model
model = HybridModel(
    n_residues=len(sequence),
    n_qubits=10,
    use_quantum=True,
    device='cuda'
)

# Predict structure
model.eval()
with torch.no_grad():
    outputs = model(sequence, n_candidates=10)

print(f"Energy: {outputs['best_energy']:.4f}")
print(f"Structure: {outputs['best_structure'].shape}")  # (10, 3)
print(f"Quantum fraction: {outputs['quantum_fraction']:.2%}")
```

### Training Your Own Model

```python
from hqpf.training import Trainer
from hqpf.data import ProteinDataset

# Load dataset
dataset = ProteinDataset('data/proteins.csv')
trainer = Trainer(model, dataset)

# Train
trainer.train(
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    checkpoint_dir='checkpoints/'
)
```

## 🏛️ Architecture

### System Overview

```
Amino Acid Sequence
       ↓
   Embedding + GNN
       ↓
  3D Coordinates  →  Ensemble Generation
       ↓                      ↓
  VQE Energy  ←  Multiple Candidates
  Evaluation              ↓
       ↓           Ranking & Selection
  Best Structure
```

### Components

1. **Structure Generator** (Classical)
   - Amino acid embeddings (20×64)
   - Positional encodings
   - Multi-layer GNN (3-6 layers)
   - 3D coordinate regression

2. **VQE Energy Solver** (Quantum)
   - Hamiltonian construction from coordinates
   - Hardware-efficient ansatz
   - Parameterized quantum circuits
   - Energy expectation value

3. **Hybrid Optimization**
   - Joint gradient descent
   - Quantum-aware backpropagation
   - Surrogate model for efficiency
   - Adaptive quantum/classical ratio

## 🧪 Features

### Quantum Computing
- ✅ VQE-based energy evaluation
- ✅ Multiple ansatz types (hardware-efficient, real amplitudes)
- ✅ Qiskit integration (Aer simulators + IBM Quantum)
- ✅ Circuit optimization
- ✅ Quantum-classical gradient flow

### Machine Learning
- ✅ Graph Neural Networks for structure
- ✅ Attention mechanisms
- ✅ Ensemble generation
- ✅ Surrogate energy models
- ✅ Mixed precision training (FP16)
- ✅ Distributed training (DDP)

### Production Features
- ✅ Comprehensive test suite (>80% coverage target)
- ✅ CI/CD with GitHub Actions
- ✅ Code quality tools (Black, Flake8, MyPy, Pylint)
- ✅ Docker containerization
- ✅ Extensive documentation
- ✅ Benchmark datasets included
- ✅ Reproducible results (seed control)

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Training Guide](docs/TRAINING.md)
- [Contributing](CONTRIBUTING.md)
- [Publication Checklist](docs/PUBLICATION_CHECKLIST.md)

## 📋 Requirements

### Minimum
- Python 3.9+
- PyTorch 2.0+
- Qiskit 1.0+
- 8GB RAM
- CPU only

### Recommended
- Python 3.10+
- PyTorch 2.3+ with CUDA 12.1
- NVIDIA GPU (8GB+ VRAM)
- 16GB RAM
- IBM Quantum account (optional)

## 🔬 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=hqpf --cov-report=html

# Run specific test suite
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/benchmarks/ -v --benchmark-only

# Run with markers
pytest -m "not slow"  # Skip slow tests
pytest -m gpu  # GPU tests only
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code quality checks
black hqpf/ tests/
isort hqpf/ tests/
flake8 hqpf/ tests/
mypy hqpf/
pylint hqpf/
```

## 🐛 Known Issues

- Quantum hardware support requires IBM Quantum account
- Very long proteins (>300 residues) may hit memory limits
- Windows support is experimental (use WSL2 recommended)

See [Issues](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/issues) for current bugs and feature requests.

## 🛣️ Roadmap

### Version 0.2.0 (Q2 2026)
- [ ] Multi-chain protein support
- [ ] Protein-protein docking
- [ ] Active learning for training
- [ ] Real quantum hardware optimization
- [ ] Web interface

### Version 0.3.0 (Q3 2026)
- [ ] Protein design capabilities
- [ ] Mutation effect prediction
- [ ] Uncertainty quantification
- [ ] Cloud deployment

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🚀 Related Projects

- [AlphaFold](https://github.com/deepmind/alphafold)
- [ESMFold](https://github.com/facebookresearch/esm)
- [RosettaFold](https://github.com/RosettaCommons/RoseTTAFold)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [Qiskit](https://github.com/Qiskit/qiskit)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ChessEngineUS/hybrid-quantum-protein-folding&type=Date)](https://star-history.com/#ChessEngineUS/hybrid-quantum-protein-folding&Date)

## 🔍 Acknowledgments

- Quantum computing infrastructure: IBM Quantum
- GPU computing: Google Colab, NVIDIA
- Protein databases: PDB
- Open-source community: PyTorch, Qiskit

---

**Status**: 🟢 Active Development | **Version**: 0.1.0 | **Last Updated**: February 8, 2026

**Maintainer**: Tommaso R. Marena | [GitHub](https://github.com/ChessEngineUS)
