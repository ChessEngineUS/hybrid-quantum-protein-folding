# Architecture Documentation

## Overview

The Hybrid Quantum-Classical Protein Folding (HQPF) framework integrates three key components:

1. **GNN Structure Generator**: Neural network that generates candidate protein structures from sequence
2. **Quantum Hamiltonian + VQE**: Learned quantum energy function evaluated via variational quantum eigensolver
3. **CNN Surrogate Model**: Fast approximation of quantum energies to reduce hardware queries

## System Architecture

```
Input: Protein Sequence
         |
         v
  [GNN Generator]
         |
         v
  Candidate Structures (x1, x2, ..., xN)
         |
         +---> [Quantum VQE] ---> E_quantum (10%)
         |           |
         |           v
         +---> [Surrogate] -----> E_approx (90%)
         |
         v
  [Energy Fusion]
         |
         v
  Best Structure + Energy
```

## Module Details

### 1. Structure Generator (`hqpf/models/structure_generator.py`)

**Architecture:**
- Input: Amino acid sequence (tensor of indices)
- Embedding: Sequence + positional encoding
- Graph construction: Backbone + long-range contact edges
- Message passing: 4 layers of graph convolutions
- Decoder: MLP → 3D coordinates or dihedral angles

**Key Features:**
- Supports both lattice (discrete) and off-lattice (continuous) representations
- Generates ensemble of structures via stochastic sampling
- Learns prior distribution over structures for regularization

**Training:**
- Supervised loss on known native structures (RMSD)
- Energy-based ranking loss
- Prior regularization

### 2. Quantum Hamiltonian (`hqpf/models/quantum_hamiltonian.py`)

**Hamiltonian Form:**

```
H_eff(θ) = H_local(θ) + H_contact(θ) + H_env(θ)
```

Where:
- `H_local`: Backbone dihedral preferences (sequence-dependent)
- `H_contact`: Non-local residue-residue interactions
- `H_env`: Environment-dependent terms (pH, ionic strength, force)

**Parameters (θ):**
- `J_i(φ)`: Local dihedral coupling (learned per residue type)
- `K_ij(θ)`: Pairwise contact strength (hydrophobicity-based)
- `α_i, β_ij`: Environment coefficients

**Qubit Encoding:**
- Direct state encoding: Each residue position → multi-qubit register
- For N=10 residues on cubic lattice: ~20-24 qubits
- Reduction via fragment-based decomposition for larger proteins

**Training:**
- Supervised regression on experimental ΔΔG, NMR ensembles
- Energy ranking loss: native < decoys
- Periodically updated (every 10 training iterations)

### 3. VQE Solver (`hqpf/models/vqe_solver.py`)

**Algorithm:**

```python
E_0 ≈ min_α <ψ(α)|H(θ)|ψ(α)>
```

**Ansatz:**
- Hardware-efficient: Alternating RY(θ) + entangling XX/YY/ZZ layers
- Problem-inspired: Respects secondary structure constraints
- Circuit depth: 3-5 layers (NISQ-compatible)

**Optimization:**
- Classical optimizer: COBYLA or Adam
- Gradient estimation: Parameter-shift rule
- Typical convergence: 50-100 VQE iterations

**Hardware Support:**
- Simulator: Qiskit Aer (fast, noiseless)
- Real backend: IBM Quantum (Falcon/Heron), IonQ Harmony
- Error mitigation: Zero-noise extrapolation, readout correction

### 4. Surrogate Model (`hqpf/models/surrogate.py`)

**Architecture:**
- Input: Conformation matrix (N_residues × features)
  - Features: AA type, coordinates, contacts, curvature
- Conv1D layers: Extract local structural patterns
- Global pooling + MLP: Scalar energy prediction
- Uncertainty head: Epistemic uncertainty for active learning

**Training:**
- Supervised on cached quantum evaluations
- MSE loss: `L = |E_surr - E_quantum|^2`
- Update frequency: Every 100 training iterations
- Cache size: 200 quantum samples

**Active Learning:**
- High-uncertainty samples prioritized for quantum evaluation
- Reduces quantum queries by 90-95%

### 5. Hybrid Model (`hqpf/models/hybrid_model.py`)

**Forward Pass:**

1. Generate N candidate structures: `{x_1, ..., x_N} = G_φ(s)`
2. Evaluate energies:
   - 10% via quantum: `E_q(x_i; θ) = VQE(H(θ, x_i))`
   - 90% via surrogate: `E_s(x_i) = Surrogate(x_i)`
3. Classical energy: `E_classical(x_i) = E_hydro + E_hbond + E_entropy`
4. Hybrid fusion: `E_total = E_classical + λ · E_quantum + ε · prior`
5. Select best: `x* = argmin E_total`

**Loss Function:**

```
L = w_1 · RMSD(x*, x_native) 
    + w_2 · L_ranking(E_total, labels) 
    + w_3 · L_prior(log p(x))
```

**Training Strategy:**
- Alternating optimization:
  1. Update generator (φ) with hybrid loss
  2. Update Hamiltonian (θ) via supervised regression (every 10 iters)
  3. Refresh surrogate with new quantum samples (every 100 iters)
- Gradient flow: PyTorch autodiff through classical + surrogate components
- Quantum gradients: Parameter-shift rule (not backprop through VQE)

## Data Pipeline

### Dataset Format

**ProteinDataset** (`hqpf/data/dataset.py`):

```json
{
  "sequence": "ACDEFGHIKLMNPQRSTVWY",
  "native_structure": [[x1, y1, z1], [x2, y2, z2], ...],
  "delta_G": -5.2,
  "stability_labels": [1, 0, 0, 1],
  "metadata": {...}
}
```

**IDRDataset**:
- Ensemble of structures (NMR-derived)
- Distance constraints (FRET, PRE)
- No single "native" structure

**BenchmarkDataset**:
- Standard proteins from CASP/CAMEO
- High-quality PDB structures
- Experimental validation data

### Data Augmentation

- Random rotations (SO(3))
- Random translations
- Gaussian noise (σ = 0.5 Å)
- Applied during training only

## Training Infrastructure

### Trainer (`hqpf/training/trainer.py`)

**Features:**
- Training loop with validation
- Checkpointing and early stopping
- Learning rate scheduling
- Logging (file + console)
- Benchmarking utilities

**Metrics:**
- RMSD to native structure
- TM-score (structural similarity)
- Energy ranking accuracy
- Ensemble diversity (for IDRs)

### Configuration

Example (`config/basic.json`):

```json
{
  "n_epochs": 50,
  "n_candidates": 10,
  "lr_generator": 1e-4,
  "lr_hamiltonian": 1e-5,
  "lr_surrogate": 1e-4,
  "validation_freq": 5,
  "surrogate_refresh_freq": 100,
  "early_stopping_patience": 20
}
```

## Hardware Requirements

### Classical Computing

- **CPU**: 8+ cores (for data processing)
- **GPU**: NVIDIA V100/A100 (16+ GB VRAM) for training
  - GNN + surrogate training
  - Batch size typically 1 (variable-length proteins)
- **RAM**: 32+ GB

### Quantum Computing

**Simulator:**
- Qiskit Aer: 20-30 qubits (exact simulation)
- PennyLane Lightning: GPU-accelerated

**Real Hardware:**
- IBM Quantum: Falcon (27 qubits), Heron (133 qubits)
- IonQ: Harmony (11 qubits), Aria (25 qubits)
- Access: Academic program or cloud credits
- Queue time: 1000 jobs/month (typical allocation)

### Estimated Resource Usage

**Training (50 epochs, 10 proteins):**
- Classical compute: ~10 GPU hours
- Quantum queries: ~500-1000 VQE runs
- QPU time: ~5-10 hours (depending on backend)

**Inference (single protein):**
- With surrogate: <1 second (GPU)
- With quantum: ~30 seconds/candidate (real hardware)

## Extension Points

### Adding New Hamiltonians

```python
from hqpf.models.quantum_hamiltonian import QuantumHamiltonian

class CustomHamiltonian(QuantumHamiltonian):
    def build_hamiltonian(self, sequence, structure):
        # Define custom Pauli operators
        ...
        return hamiltonian_matrix
```

### Custom Datasets

```python
from hqpf.data import ProteinDataset

class MyDataset(ProteinDataset):
    def _load_data(self):
        # Load from custom source
        ...
        return data_list
```

### New Architectures

- Replace GNN with diffusion model
- Add attention mechanisms
- Multi-scale coarse-graining
- Integrate AlphaFold embeddings

## Performance Optimization

### Tips for Faster Training

1. **Use surrogate early**: Start with 95% surrogate, reduce to 90% after convergence
2. **Cache quantum results**: Reuse VQE results for similar structures
3. **Fragment-based**: Decompose large proteins (N>20) into overlapping fragments
4. **Mixed precision**: Use `torch.cuda.amp` for faster GPU training
5. **Parallel quantum**: Run multiple VQE jobs in parallel (if quota allows)

### Debugging

- Set `use_quantum=False` to use simulator only
- Reduce `n_candidates` to 5 for faster iteration
- Check surrogate accuracy: `loss < 0.1` (normalized energy units)
- Monitor quantum convergence: VQE should converge in <100 iters

## Citation

If you use this code, please cite:

```bibtex
@software{hqpf2026,
  author = {Marena, Tommaso R.},
  title = {Hybrid Quantum-Classical Protein Folding},
  year = {2026},
  url = {https://github.com/ChessEngineUS/hybrid-quantum-protein-folding}
}
```

## References

1. Jumper et al. (2021). AlphaFold2. Nature.
2. Peruzzo et al. (2014). VQE. Nature Communications.
3. Cao et al. (2023). Quantum chemistry with VQE. Nature Reviews Physics.
4. Fingerhuth et al. (2018). Quantum protein folding. arXiv.
5. Baker & Hubbard (2006). Rosetta. Current Opinion in Structural Biology.
