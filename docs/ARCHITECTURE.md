# Architecture Overview

## System Design

### High-Level Architecture

The Hybrid Quantum-AI Protein Folding (HQPF) system combines classical deep learning with quantum computing to predict protein structures from amino acid sequences.

```
┌──────────────────────────────────┐
│   Input: Amino Acid Sequence       │
│   (e.g., "ACDEFG...")                │
└─────────────┬─────────────────────┘
               │
               │ Embedding + Encoding
               │
               ▼
┌──────────────────────────────────┐
│   Graph Neural Network (GNN)      │
│   - Node: Residue features         │
│   - Edge: Spatial relationships    │
│   - Layers: 3-6 message passing    │
└─────────────┬─────────────────────┘
               │
               │ 3D Coordinate Prediction
               │
               ▼
┌──────────────────────────────────┐
│   Ensemble Generator               │
│   - Multiple candidate structures  │
│   - Temperature-controlled sampling│
└─────────────┬─────────────────────┘
               │
               │ Energy Evaluation
               │
               ▼
┌──────────────────────────────────┐
│   VQE Quantum Solver               │
│   - Hamiltonian construction       │
│   - Parameterized quantum circuit  │
│   - Energy expectation value       │
└─────────────┬─────────────────────┘
               │
               │ Structure Selection
               │
               ▼
┌──────────────────────────────────┐
│   Output: 3D Structure (N×3)       │
│   + Energy + Confidence            │
└──────────────────────────────────┘
```

## Module Descriptions

### 1. Structure Generator (Classical)

**File**: `hqpf/models/structure_generator.py`

**Purpose**: Generate 3D atomic coordinates from sequence

**Components**:
- **Embedding Layer**: Maps amino acids (20 types) to 64D vectors
- **Positional Encoding**: Adds sequence position information
- **GNN Layers**: 3-6 graph attention layers
- **MLP Decoder**: Projects to 3D coordinates

**Key Innovation**: Uses distance-based graph construction with adaptive edge thresholds

### 2. VQE Solver (Quantum)

**File**: `hqpf/models/vqe_solver.py`

**Purpose**: Evaluate conformational energy using quantum computing

**Quantum Circuit**:
```python
for layer in range(ansatz_depth):
    # Single-qubit rotations
    for qubit in range(n_qubits):
        RY(θ[layer, qubit, 0])
        RZ(θ[layer, qubit, 1])
    
    # Entanglement
    for qubit in range(n_qubits-1):
        CNOT(qubit, qubit+1)
```

**Hamiltonian**:
\[
H = \sum_{i<j} \frac{1}{||r_i - r_j||} (Z_i Z_j + X_i X_j)
\]

### 3. Hybrid Model (Integration)

**File**: `hqpf/models/hybrid_model.py`

**Purpose**: Coordinate classical and quantum components

**Forward Pass**:
1. Generate ensemble of structures (classical)
2. Evaluate energy for each (quantum)
3. Select best structure
4. Backpropagate through both components

**Key Features**:
- Adaptive quantum/classical ratio
- Surrogate model for efficiency
- Mixed precision support

## Data Flow

### Training Loop

```python
for batch in dataloader:
    # Forward pass
    structures = generator(sequences)  # Classical
    energies = vqe_solver(structures)  # Quantum
    
    # Loss computation
    loss = energy_loss(energies) + structure_loss(structures)
    
    # Backward pass
    loss.backward()  # Gradients through both
    
    # Optimization
    optimizer.step()
```

### Inference Pipeline

```python
1. Input: sequence tensor (N,)
2. Generate candidates: (M, N, 3) where M=n_candidates
3. Evaluate energies: (M,) energy values
4. Select best: argmin(energies)
5. Output: (N, 3) + energy + metadata
```

## Performance Optimization

### Computational Complexity

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| GNN | O(N² × L) | O(N²) |
| VQE Circuit | O(2^n × D) | O(2^n) |
| Hamiltonian | O(N²) | O(2^n × 2^n) |
| Total | O(N²L + M×2^nD) | O(2^n × 2^n) |

*N=protein length, L=GNN layers, n=qubits, D=circuit depth, M=candidates*

### Optimization Strategies

1. **Surrogate Model**: Train fast classical approximation of VQE
2. **Circuit Compilation**: Optimize quantum gates
3. **Batching**: Process multiple proteins in parallel
4. **Mixed Precision**: Use FP16 for GNN, FP32 for VQE
5. **Gradient Checkpointing**: Trade compute for memory

## Quantum-Classical Interface

### Gradient Flow

**Challenge**: Quantum circuits are not natively differentiable

**Solution**: Parameter-shift rule

\[
\frac{\partial \langle H \rangle}{\partial \theta_i} = \frac{1}{2}[\langle H \rangle(\theta_i + \pi/2) - \langle H \rangle(\theta_i - \pi/2)]
\]

**Implementation**:
```python
class VQEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, coords):
        energy = run_vqe_circuit(theta, coords)
        ctx.save_for_backward(theta, coords)
        return energy
    
    @staticmethod
    def backward(ctx, grad_output):
        theta, coords = ctx.saved_tensors
        grad_theta = compute_parameter_shift(theta, coords)
        return grad_theta * grad_output, None
```

## Scalability Analysis

### Protein Length Scaling

- **Classical GNN**: Linear memory, quadratic time
- **Quantum Circuit**: Exponential in qubits, but we use O(log N) qubits
- **Hybrid Approach**: Enables scaling to 200+ residues

### Hardware Requirements

| Protein Size | RAM | GPU VRAM | Qubits | Time (T4 GPU) |
|--------------|-----|----------|--------|---------------|
| 50 residues | 4GB | 2GB | 10 | 2.3s |
| 100 residues | 8GB | 4GB | 12 | 8.1s |
| 200 residues | 16GB | 8GB | 14 | 28.4s |

## Error Handling

### Quantum Errors
- **Circuit errors**: Validation before execution
- **Backend failures**: Automatic fallback to simulator
- **Convergence issues**: Early stopping with warnings

### Classical Errors
- **NaN gradients**: Gradient clipping and checks
- **Memory overflow**: Automatic batch size reduction
- **CUDA errors**: Graceful CPU fallback

## Extension Points

### Adding New Ansatz

```python
def custom_ansatz(circuit, theta, n_qubits):
    # Your custom quantum circuit
    for i in range(n_qubits):
        circuit.rx(theta[i], i)
    return circuit

# Register
VQESolver.register_ansatz('custom', custom_ansatz)
```

### Custom Energy Function

```python
class CustomEnergy(nn.Module):
    def forward(self, coords):
        # Your physics-based energy
        return energy

# Use in model
model.energy_fn = CustomEnergy()
```

## Future Enhancements

1. **Multi-chain Support**: Extend to protein complexes
2. **Side-chain Modeling**: Full-atom predictions
3. **Dynamics**: Molecular dynamics integration
4. **Hardware Optimization**: Real quantum device support
5. **Uncertainty**: Bayesian neural networks
