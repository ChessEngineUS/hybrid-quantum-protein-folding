# Quickstart Guide

Get started with hybrid quantum-classical protein folding in 5 minutes!

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ChessEngineUS/hybrid-quantum-protein-folding.git
cd hybrid-quantum-protein-folding
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the package

```bash
pip install -e .
```

## Quick Examples

### Example 1: Basic Structure Prediction

```python
import torch
from hqpf.models import HybridModel

# Define a protein sequence
sequence_str = "ACDEFGHIKLMNPQRSTVWY"

# Convert to tensor
aa_to_idx = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}
sequence = torch.tensor([aa_to_idx[aa] for aa in sequence_str], dtype=torch.long)

# Initialize model
model = HybridModel(
    n_residues=len(sequence),
    n_qubits=20,
    use_quantum=False,  # Use simulator
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

### Example 2: Training on Custom Data

```python
from hqpf.models import HybridModel
from hqpf.data import ProteinDataset
from hqpf.training import Trainer

# Create datasets
train_dataset = ProteinDataset(
    data_dir='./data',
    split='train',
    max_length=25
)

val_dataset = ProteinDataset(
    data_dir='./data',
    split='val',
    max_length=25
)

# Initialize model
model = HybridModel(
    n_residues=25,
    n_qubits=20,
    use_quantum=False,
    device='cuda'  # Use GPU if available
)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config={
        'n_epochs': 50,
        'n_candidates': 10,
        'checkpoint_dir': './checkpoints'
    }
)

# Train
metrics = trainer.train()
```

### Example 3: Inference from Checkpoint

```python
import torch
from hqpf.models import HybridModel

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Initialize model
model = HybridModel(
    n_residues=20,
    n_qubits=20,
    use_quantum=False,
    device='cpu'
)

# Load weights
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Predict
sequence = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                         10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

with torch.no_grad():
    outputs = model(sequence, n_candidates=20)

print(f"Best structure: {outputs['best_structure']}")
```

## Command-Line Usage

### Training

```bash
# Basic training
python examples/train_basic.py \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --n_epochs 50

# With custom config
python examples/train_basic.py \
    --config config/basic.json

# Using quantum hardware (requires IBM account)
python examples/train_basic.py \
    --config config/quantum_hardware.json \
    --use_quantum
```

### Inference

```bash
# Predict structure
python examples/inference_demo.py \
    --sequence ACDEFGHIKLMNPQRSTVWY \
    --checkpoint checkpoints/best_model.pt \
    --output structure.pdb

# Visualize (requires matplotlib)
python examples/inference_demo.py \
    --sequence ACDEFGHIKLMNPQRSTVWY \
    --checkpoint checkpoints/best_model.pt \
    --output structure.png
```

### Benchmarking

```bash
# Benchmark on IDRs
python examples/benchmark_idrs.py \
    --checkpoint checkpoints/best_model.pt \
    --output_dir ./benchmark_results
```

## Interactive Tutorial

Run the Jupyter notebook tutorial:

```bash
jupyter notebook notebooks/tutorial.ipynb
```

This covers:
- Loading and preprocessing data
- Model architecture details
- Training and validation
- Structure visualization
- Energy landscape analysis

## Working with Real Quantum Hardware

### 1. Setup IBM Quantum Account

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your credentials
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN"
)
```

### 2. Run with Quantum Backend

```python
from hqpf.models import HybridModel

model = HybridModel(
    n_residues=20,
    n_qubits=20,
    use_quantum=True,
    backend='ibm',  # or 'ionq'
    device='cpu'
)

# The model will automatically use real quantum hardware
# for a fraction of energy evaluations
outputs = model(sequence, n_candidates=10)
```

### 3. Monitor Quantum Usage

```python
print(f"Quantum hardware usage: {outputs['quantum_fraction']*100:.1f}%")
```

## Tips for Best Results

### 1. Start with Small Proteins

- Begin with sequences of 10-20 residues
- Increase gradually as you optimize hyperparameters
- Use fragment-based approach for N>25

### 2. Tune Hyperparameters

Key parameters to adjust:

```python
config = {
    'n_candidates': 10,          # More = better, but slower
    'lr_generator': 1e-4,        # Adjust if training unstable
    'surrogate_refresh_freq': 100,  # More frequent = better surrogate
    'quantum_fraction': 0.1      # Higher = more accurate, but expensive
}
```

### 3. Use Proper Validation

- Always use a held-out validation set
- Monitor RMSD and energy ranking metrics
- Early stopping prevents overfitting

### 4. Leverage Pre-trained Models

```python
# Load pre-trained weights
checkpoint = torch.load('pretrained_models/idr_model.pt')
model.load_state_dict(checkpoint['model_state'])

# Fine-tune on your data
trainer = Trainer(model, your_dataset)
trainer.train()
```

## Common Issues

### Issue 1: CUDA Out of Memory

**Solution:** Reduce batch size or use gradient checkpointing

```python
config['batch_size'] = 1  # Already default
config['n_candidates'] = 5  # Reduce candidates
```

### Issue 2: Surrogate Diverges

**Solution:** Increase surrogate refresh frequency

```python
config['surrogate_refresh_freq'] = 50  # More frequent updates
```

### Issue 3: VQE Doesn't Converge

**Solution:** Adjust VQE parameters

```python
model = HybridModel(
    ...,
    n_vqe_layers=5,  # Increase ansatz depth
)
```

### Issue 4: Quantum Backend Queue Too Long

**Solution:** Use simulator during development

```python
model = HybridModel(
    ...,
    use_quantum=False,  # Use fast simulator
)
```

## Next Steps

- Read [Architecture Documentation](ARCHITECTURE.md) for detailed design
- Check [examples/](../examples/) for more advanced usage
- Join discussions on GitHub Issues
- Contribute improvements via Pull Requests

## Getting Help

- **Documentation:** See [docs/](../docs/) directory
- **Issues:** [GitHub Issues](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/discussions)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hqpf2026,
  author = {Marena, Tommaso R.},
  title = {Hybrid Quantum-Classical Protein Folding},
  year = {2026},
  url = {https://github.com/ChessEngineUS/hybrid-quantum-protein-folding}
}
```
