# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial implementation of hybrid quantum-classical protein folding framework
- VQE-based quantum Hamiltonian evaluation
- GNN structure generator with learned priors
- CNN surrogate model for reduced quantum hardware usage
- Fragment-based approach for proteins >20 residues
- Hardware-efficient ansatz for NISQ devices
- Support for IBM Quantum and IonQ backends
- Coarse-grained lattice representation
- Classical energy terms (hydrophobic, H-bonding, entropy)
- End-to-end training with hybrid loss function
- Benchmark datasets for IDRs and metamorphic proteins
- Example scripts for training and inference
- Comprehensive documentation (README, QUICKSTART, ARCHITECTURE)
- Unit tests for core modules
- Configuration management system

### Infrastructure
- GitHub repository setup
- Requirements and setup.py for installation
- Pre-commit hooks for code quality
- CI/CD pipeline (planned)
- Example notebooks for tutorials

## [0.1.0] - 2026-02-07

### Added
- Initial project structure
- Core module implementations:
  - `hqpf.models.HybridModel`: Main hybrid quantum-classical model
  - `hqpf.models.QuantumHamiltonian`: Parametric learned Hamiltonian
  - `hqpf.models.VQESolver`: VQE implementation with parameter-shift gradients
  - `hqpf.models.StructureGenerator`: GNN-based structure decoder
  - `hqpf.models.SurrogateModel`: CNN surrogate for energy prediction
  - `hqpf.data.CoarseGrainedEncoding`: Lattice and dihedral encodings
  - `hqpf.data.BenchmarkDatasets`: Target problem datasets
  - `hqpf.training.Trainer`: Training loop with surrogate refresh
  - `hqpf.training.HybridLoss`: Combined structural and energy ranking loss
  - `hqpf.quantum.circuits`: Quantum circuit construction
  - `hqpf.quantum.ansatz`: Hardware-efficient and problem-inspired ansätze
  - `hqpf.quantum.backends`: Backend management for IBM/IonQ/simulators
  - `hqpf.quantum.error_mitigation`: ZNE and readout correction
  - `hqpf.classical.energy_terms`: Classical potential functions
  - `hqpf.utils.metrics`: RMSD, TM-score, ΔΔG calculations
  - `hqpf.utils.visualization`: Structure and energy landscape plotting

- Documentation:
  - README.md with project overview and quick start
  - docs/QUICKSTART.md with detailed usage examples
  - docs/ARCHITECTURE.md with system design details
  - CONTRIBUTING.md with contribution guidelines
  - LICENSE (MIT)

- Example scripts:
  - `examples/train_basic.py`: Basic training example
  - `examples/inference_demo.py`: Structure prediction from checkpoint
  - `examples/benchmark_idrs.py`: IDR benchmarking pipeline

- Configuration:
  - `config/default_config.yaml`: Default hyperparameters
  - Support for custom config files

### Technical Details

- **Quantum Backend Support**:
  - IBM Quantum (Falcon 27-qubit, Heron 133-qubit)
  - IonQ (11-88 qubits)
  - PennyLane simulators for development

- **Model Architecture**:
  - Coarse-grained representation: 3.8 Å lattice spacing
  - Quantum encoding: 20-30 qubits for N=10-20 residues
  - VQE ansatz depth: 3-5 layers (configurable)
  - GNN: 3-layer graph network with 128 hidden dimensions
  - Surrogate: 3-layer CNN with 32-64 feature maps

- **Training**:
  - Hybrid loss: structural + energy ranking + learned prior
  - Surrogate refresh: every 100 iterations (default)
  - Quantum fraction: 10% of energy evaluations (configurable)
  - Optimizer: Adam with learning rate 1e-4
  - Gradient estimation: Parameter-shift rule for quantum gradients

- **Benchmarks**:
  - Problem Set A: p53-TAD, c-Fos linker (IDRs)
  - Problem Set B: FF domain, p53 DBD mutants (metamorphic)
  - Problem Set C: Zika virus fragment (constrained)

### Dependencies

- Python 3.10+
- PyTorch 2.2+
- Qiskit 1.0+
- PennyLane 0.35+
- NumPy, SciPy, pandas
- matplotlib for visualization
- BioPython for structure handling

### Known Limitations

- Scalability limited to ~25 residues without fragment-based approach
- Quantum hardware availability dependent on provider queues
- Surrogate model accuracy degrades without periodic refresh
- Classical baseline comparisons require external tools (AlphaFold2, Rosetta)

### Planned Features

- [ ] Pre-trained models for common protein families
- [ ] Interactive visualization dashboard
- [ ] Support for additional quantum backends (Rigetti, Xanadu)
- [ ] Automated hyperparameter tuning
- [ ] Distributed training across multiple GPUs
- [ ] Real-time monitoring of quantum circuit execution
- [ ] Integration with PDB and UniProt databases
- [ ] Automated benchmarking against CASP/CAMEO targets
- [ ] Fragment assembly with message-passing interface
- [ ] Support for pH, ionic strength, and mechanical force conditions

## Version History

### Version Numbering

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes and minor improvements

### Release Schedule

- **Alpha (0.x.x)**: Initial development, API unstable
- **Beta (1.0.0-beta.x)**: Feature complete, testing phase
- **Stable (1.0.0+)**: Production-ready, API stable

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{hqpf2026,
  author = {Marena, Tommaso R.},
  title = {Hybrid Quantum-AI Energy Fusion for Protein Structure Prediction},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/ChessEngineUS/hybrid-quantum-protein-folding},
  doi = {TBD}
}
```

## Acknowledgments

- IBM Quantum for quantum hardware access
- Baker Lab (IPD) for protein design insights and methodologies
- Aspuru-Guzik group for quantum chemistry frameworks
- The open-source community for foundational tools (PyTorch, Qiskit, PennyLane)

---

*For detailed changes in each commit, see the [commit history](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/commits/main).*
