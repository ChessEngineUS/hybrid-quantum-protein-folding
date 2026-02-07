# Getting Started with Hybrid Quantum-AI Protein Folding

Welcome! This guide will walk you through your first experience with hybrid quantum-classical protein structure prediction, from installation to running your first predictions.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [First Steps](#first-steps)
4. [Understanding the Output](#understanding-the-output)
5. [Next Steps](#next-steps)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Knowledge Prerequisites

- **Python programming**: Comfortable with basic Python (functions, classes, imports)
- **Machine learning basics**: Understanding of neural networks, training loops
- **Quantum computing**: Helpful but not required (we'll explain as we go)
- **Protein biology**: Basic understanding of protein sequences and structures

If you're missing any of these, don't worry! The code is designed to be accessible, and we provide links to resources.

### System Requirements

**Minimum:**
- Python 3.10 or higher
- 8GB RAM
- 10GB disk space
- CPU: Modern multi-core processor

**Recommended:**
- Python 3.11
- 32GB RAM
- 50GB disk space
- GPU: NVIDIA with CUDA support (for faster training)
- Fast internet (for quantum backend access)

### Software Prerequisites

1. **Python**: Check your version
   ```bash
   python --version  # Should be 3.10 or higher
   ```

2. **pip**: Python package installer
   ```bash
   pip --version
   ```

3. **Git**: Version control
   ```bash
   git --version
   ```

4. **(Optional) CUDA**: For GPU acceleration
   ```bash
   nvidia-smi  # Check if CUDA is available
   ```

If any are missing, see [Installation Help](#installation-help) below.

## Installation

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/ChessEngineUS/hybrid-quantum-protein-folding.git
cd hybrid-quantum-protein-folding
```

You should see:
```
Cloning into 'hybrid-quantum-protein-folding'...
remote: Enumerating objects: ...
```

### Step 2: Create a Virtual Environment (Recommended)

This isolates the project dependencies:

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will take a few minutes. You'll see:
```
Collecting torch>=2.2.0
Collecting qiskit>=1.0.0
...
Successfully installed ...
```

### Step 4: Install the Package

```bash
pip install -e .
```

The `-e` flag installs in "editable" mode, meaning changes to the code take effect immediately.

### Step 5: Verify Installation

Test that everything works:

```bash
python -c "import hqpf; print('Installation successful!')"
```

You should see:
```
Installation successful!
```

If you see an error, jump to [Troubleshooting](#troubleshooting).

## First Steps

### Your First Structure Prediction

Let's predict the structure of a simple 10-amino acid peptide.

#### Create a Test Script

Create a file called `my_first_prediction.py`:

```python
# my_first_prediction.py
import torch
from hqpf.models import HybridModel

# Define a simple protein sequence (10 amino acids)
sequence_str = "ACDEFGHIKL"

# Convert amino acids to numerical indices
aa_to_idx = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

sequence = torch.tensor(
    [aa_to_idx[aa] for aa in sequence_str],
    dtype=torch.long
)

print(f"Sequence: {sequence_str}")
print(f"Sequence tensor: {sequence}")
print(f"Length: {len(sequence)} residues")

# Initialize the model
print("\nInitializing hybrid quantum-classical model...")
model = HybridModel(
    n_residues=len(sequence),
    n_qubits=10,           # One qubit per residue
    use_quantum=False,     # Use classical simulator for now
    device='cpu'           # Use CPU (change to 'cuda' if you have GPU)
)

print("Model initialized successfully!")
print(f"Parameters: {sum(p.numel() for p in model.parameters())} total")

# Predict structure
print("\nPredicting structure...")
model.eval()  # Set to evaluation mode

with torch.no_grad():  # No gradient computation needed
    outputs = model(sequence, n_candidates=5)

print("\nPrediction complete!")

# Extract results
best_structure = outputs['best_structure']
best_energy = outputs['best_energy']

print(f"\nResults:")
print(f"  Best structure shape: {best_structure.shape}")
print(f"  Best energy: {best_energy.item():.4f}")
print(f"  First 3 residue coordinates:")
print(best_structure[:3])

print("\n✅ Success! You've predicted your first protein structure.")
```

#### Run It

```bash
python my_first_prediction.py
```

#### What You Should See

```
Sequence: ACDEFGHIKL
Sequence tensor: tensor([0, 4, 3, 6, 13, 7, 8, 9, 11, 10])
Length: 10 residues

Initializing hybrid quantum-classical model...
Model initialized successfully!
Parameters: 127543 total

Predicting structure...
Prediction complete!

Results:
  Best structure shape: torch.Size([10, 3])
  Best energy: -12.3456
  First 3 residue coordinates:
  tensor([[ 0.1234,  0.5678, -0.9012],
          [ 1.2345, -0.3456,  0.7890],
          [-0.4567,  1.8901,  0.2345]])

✅ Success! You've predicted your first protein structure.
```

### What Just Happened?

1. **Sequence Encoding**: We converted the amino acid sequence `ACDEFGHIKL` into numbers
2. **Model Initialization**: Created a hybrid model with:
   - Classical neural network (GNN structure generator)
   - Quantum circuit simulator (no real quantum hardware yet)
   - Energy evaluation components
3. **Structure Prediction**: The model:
   - Generated 5 candidate structures
   - Evaluated their energies (classical + quantum)
   - Returned the lowest-energy structure
4. **Output**: Got 10 sets of (x, y, z) coordinates, one per residue

### Visualize the Structure

Add this to your script:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Convert to numpy for plotting
structure_np = best_structure.cpu().numpy()

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot backbone trace
ax.plot(structure_np[:, 0], structure_np[:, 1], structure_np[:, 2],
        'b-', linewidth=2, label='Backbone')

# Plot residues as spheres
ax.scatter(structure_np[:, 0], structure_np[:, 1], structure_np[:, 2],
          c=range(len(sequence)), cmap='rainbow', s=100, alpha=0.8)

# Add residue labels
for i, (x, y, z) in enumerate(structure_np):
    ax.text(x, y, z, sequence_str[i], fontsize=12)

ax.set_xlabel('X (Å)')
ax.set_ylabel('Y (Å)')
ax.set_zlabel('Z (Å)')
ax.set_title(f'Predicted Structure: {sequence_str}')
ax.legend()

plt.savefig('my_first_structure.png', dpi=300, bbox_inches='tight')
print("\nStructure saved to: my_first_structure.png")
plt.show()
```

Run again:
```bash
python my_first_prediction.py
```

You'll see a 3D visualization of your predicted protein structure!

## Understanding the Output

### Structure Coordinates

The output `best_structure` is a tensor of shape `(N, 3)` where:
- `N` = number of residues (10 in our example)
- `3` = x, y, z coordinates in Ångströms

Each row is the position of one Cα (alpha carbon) atom.

### Energy Score

The `best_energy` is the total energy:
```
E_total = E_classical + λ * E_quantum + E_prior
```

Where:
- **E_classical**: Hydrophobic interactions, H-bonds, entropy
- **E_quantum**: Learned quantum Hamiltonian contribution
- **E_prior**: GNN-learned structural preferences
- **λ**: Weighting factor (default 0.1)

Lower energy = more stable structure.

### Model Outputs Dictionary

```python
outputs = {
    'best_structure': torch.Tensor,     # Shape (N, 3)
    'best_energy': torch.Tensor,        # Scalar
    'all_structures': torch.Tensor,     # Shape (n_candidates, N, 3)
    'all_energies': torch.Tensor,       # Shape (n_candidates,)
    'quantum_fraction': float,          # Fraction using real quantum
    'surrogate_used': bool              # Whether surrogate was used
}
```

## Next Steps

### 1. Train on Your Own Data

See `examples/train_basic.py` for a complete training example:

```bash
python examples/train_basic.py --help
```

### 2. Use Real Quantum Hardware

Setup IBM Quantum account:

1. Register at [https://quantum.ibm.com](https://quantum.ibm.com)
2. Get your API token
3. Save credentials:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_TOKEN_HERE"
)
```

4. Run with quantum backend:

```python
model = HybridModel(
    n_residues=10,
    n_qubits=10,
    use_quantum=True,    # ← Enable real quantum
    backend='ibm',
    device='cpu'
)
```

### 3. Benchmark Against AlphaFold

Compare predictions on real proteins:

```bash
python examples/benchmark_idrs.py \
    --checkpoint checkpoints/model.pt \
    --compare alphafold
```

### 4. Explore Notebooks

Interactive tutorials:

```bash
jupyter notebook notebooks/tutorial.ipynb
```

### 5. Read Advanced Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design details
- [QUICKSTART.md](QUICKSTART.md) - Comprehensive usage guide
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute

## Troubleshooting

### Installation Issues

#### Problem: "Command 'python' not found"

**Solution**: Try `python3` instead:
```bash
python3 --version
python3 -m venv venv
```

#### Problem: "No module named 'torch'"

**Solution**: Ensure virtual environment is activated:
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### Problem: CUDA/GPU errors

**Solution**: Install CPU-only PyTorch:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Runtime Issues

#### Problem: "RuntimeError: CUDA out of memory"

**Solution**: Use CPU or reduce batch size:
```python
model = HybridModel(..., device='cpu')
# or
trainer = Trainer(..., config={'n_candidates': 5})  # Reduce from 10
```

#### Problem: Quantum backend queue too long

**Solution**: Use simulator:
```python
model = HybridModel(..., use_quantum=False)
```

#### Problem: Poor prediction quality

**Solutions**:
1. Train longer: `trainer.train(n_epochs=100)`
2. Use more candidates: `model(..., n_candidates=20)`
3. Refresh surrogate more often: `surrogate_refresh_freq=50`

### Getting Help

1. **Check documentation**: [docs/](../docs/)
2. **Search issues**: [GitHub Issues](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/issues)
3. **Ask questions**: [GitHub Discussions](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/discussions)
4. **Report bugs**: [New Issue](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/issues/new)

## Installation Help

### Installing Python

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3-pip python3-venv
```

**Mac:**
```bash
brew install python@3.11
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/) and run installer.

### Installing Git

**Linux:**
```bash
sudo apt install git
```

**Mac:**
```bash
brew install git
```

**Windows:**
Download from [git-scm.com](https://git-scm.com/download/win).

### Installing CUDA (Optional)

For GPU acceleration:

1. Check if you have NVIDIA GPU: `nvidia-smi`
2. Download CUDA from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
3. Install PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Quick Reference

### Common Commands

```bash
# Activate environment
source venv/bin/activate

# Train model
python examples/train_basic.py

# Run inference
python examples/inference_demo.py --sequence ACDEFG

# Run tests
pytest tests/

# Start Jupyter
jupyter notebook
```

### Important Paths

```
hqpf/                    # Main package
  models/                # Model implementations
  data/                  # Data loading
  training/              # Training code
  quantum/               # Quantum circuits
examples/                # Example scripts
notebooks/               # Jupyter tutorials
docs/                    # Documentation
config/                  # Configuration files
checkpoints/             # Saved models (created during training)
```

### Key Configuration Parameters

```python
config = {
    'n_residues': 20,              # Protein length
    'n_qubits': 20,                # Quantum circuit size
    'use_quantum': False,          # True = real hardware
    'n_candidates': 10,            # Structure samples per prediction
    'n_epochs': 50,                # Training iterations
    'lr_generator': 1e-4,          # Learning rate
    'surrogate_refresh_freq': 100, # Surrogate update frequency
    'quantum_fraction': 0.1        # % of quantum evaluations
}
```

## Learning Resources

### Quantum Computing
- [Qiskit Textbook](https://qiskit.org/textbook)
- [PennyLane Tutorials](https://pennylane.ai/qml/)
- [IBM Quantum Learning](https://learning.quantum.ibm.com/)

### Protein Structure
- [Protein Data Bank](https://www.rcsb.org/)
- [Introduction to Protein Structure](https://www.ncbi.nlm.nih.gov/books/NBK26830/)

### Machine Learning
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## Congratulations!

You've completed the getting started guide! You're now ready to:

- ✅ Predict protein structures
- ✅ Train custom models
- ✅ Use quantum hardware
- ✅ Benchmark against baselines

Happy folding! 🧬⚛️🚀

---

**Need help?** Don't hesitate to ask in [Discussions](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/discussions)!
