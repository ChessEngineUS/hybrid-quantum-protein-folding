# API Reference

## Core Models

### HybridModel

Main model class combining GNN and VQE.

```python
from hqpf.models import HybridModel

model = HybridModel(
    n_residues=100,
    n_qubits=12,
    embedding_dim=64,
    hidden_dim=128,
    n_gnn_layers=3,
    n_vqe_layers=2,
    use_quantum=True,
    backend='qasm_simulator',
    device='cuda'
)
```

**Parameters**:
- `n_residues` (int): Number of amino acids in sequence
- `n_qubits` (int): Number of qubits for VQE
- `embedding_dim` (int): Dimension of amino acid embeddings
- `hidden_dim` (int): Hidden layer dimension for GNN
- `n_gnn_layers` (int): Number of GNN layers (3-6 recommended)
- `n_vqe_layers` (int): Depth of quantum ansatz (2-4 recommended)
- `use_quantum` (bool): Enable quantum evaluation (False for testing)
- `backend` (str): Qiskit backend ('statevector_simulator', 'qasm_simulator', 'ibmq_*')
- `device` (str): PyTorch device ('cpu', 'cuda', 'cuda:0', etc.)

**Methods**:

#### `forward(sequence, n_candidates=10, use_surrogate=False, temperature=0.1)`

Predict protein structure from sequence.

**Args**:
- `sequence` (torch.Tensor): Integer tensor of shape (n_residues,), values 0-19
- `n_candidates` (int): Number of candidate structures to generate
- `use_surrogate` (bool): Use fast surrogate model instead of VQE
- `temperature` (float): Sampling temperature for diversity (0.0-1.0)

**Returns**:
```python
{
    'best_structure': torch.Tensor,  # (n_residues, 3)
    'best_energy': torch.Tensor,     # scalar
    'all_structures': List[torch.Tensor],  # n_candidates items
    'all_energies': List[torch.Tensor],    # n_candidates items
    'quantum_fraction': float,       # 0.0-1.0
    'convergence': Dict,             # VQE convergence info
}
```

### VQESolver

Quantum energy evaluator.

```python
from hqpf.models.vqe_solver import VQESolver

solver = VQESolver(
    n_qubits=10,
    ansatz_type='hardware_efficient',
    ansatz_depth=2,
    backend='statevector_simulator',
    shots=1024,
    use_error_mitigation=False
)
```

**Parameters**:
- `n_qubits` (int): Number of qubits
- `ansatz_type` (str): Circuit type ('hardware_efficient', 'real_amplitudes')
- `ansatz_depth` (int): Number of ansatz repetitions
- `backend` (str): Qiskit backend
- `shots` (int): Number of measurement shots (for QASM)
- `use_error_mitigation` (bool): Enable error mitigation (experimental)

**Methods**:

#### `forward(coords)`

Compute energy for given coordinates.

**Args**:
- `coords` (torch.Tensor): Shape (n_residues, 3)

**Returns**:
- `energy` (torch.Tensor): Scalar energy value

### StructureGenerator

GNN-based structure generator.

```python
from hqpf.models.structure_generator import StructureGenerator

generator = StructureGenerator(
    n_residues=100,
    embedding_dim=64,
    hidden_dim=128,
    n_layers=3,
    dropout=0.1,
    device='cuda'
)
```

**Methods**:

#### `generate_ensemble(sequence, n_samples=10, temperature=0.1)`

Generate multiple candidate structures.

**Args**:
- `sequence` (torch.Tensor): Shape (n_residues,)
- `n_samples` (int): Number of structures to generate
- `temperature` (float): Sampling temperature

**Returns**:
- `structures` (List[torch.Tensor]): List of (n_residues, 3) tensors
- `priors` (List[torch.Tensor]): Optional prior distributions

## Data Utilities

### Benchmark Datasets

```python
from hqpf.data.benchmarks import (
    load_benchmark_protein,
    list_benchmark_proteins,
    get_benchmark_statistics,
    AA_TO_IDX
)

# List available proteins
protein_ids = list_benchmark_proteins(dataset='test')
# Returns: ['peptide_10', 'helix_12', 'sheet_10', 'mixed_20']

# Load a protein
protein = load_benchmark_protein('peptide_10', dataset='test')
print(protein.sequence)  # "ACDEFGHIKL"
print(protein.secondary_structure)  # "CCCCCCCCCC"

# Convert to tensor
sequence = protein.to_tensor(AA_TO_IDX, device='cuda')
```

**Datasets**:
- `'test'`: Small test proteins (10-20 residues)
- `'mini'`: Mini-proteins (20-40 residues) with PDB IDs
- `'stability'`: Proteins with experimental stability data

## Training

### Trainer

```python
from hqpf.training import Trainer
from hqpf.data import ProteinDataset

# Load data
dataset = ProteinDataset('data/proteins.csv')

# Create trainer
trainer = Trainer(
    model=model,
    dataset=dataset,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda'
)

# Train
trainer.train(
    epochs=100,
    validation_split=0.2,
    checkpoint_dir='checkpoints/',
    log_interval=10
)
```

## Visualization

```python
from hqpf.visualization import (
    plot_structure_3d,
    plot_contact_map,
    plot_energy_landscape,
    plot_ramachandran
)

# 3D structure
fig = plot_structure_3d(structure, sequence_str)
fig.savefig('structure.png')

# Contact map
contacts = compute_contact_map(structure, threshold=8.0)
plot_contact_map(contacts)

# Energy landscape
plot_energy_landscape(all_structures, all_energies)
```

## Metrics

```python
from hqpf.metrics import (
    compute_rmsd,
    compute_tm_score,
    compute_gdt_ts,
    compute_contact_accuracy
)

# RMSD
rmsd = compute_rmsd(predicted, native)

# TM-score (0-1, higher better)
tm_score = compute_tm_score(predicted, native)

# GDT-TS
gdt = compute_gdt_ts(predicted, native)

# Contact prediction
contact_acc = compute_contact_accuracy(
    predicted_contacts, 
    native_contacts
)
```

## Configuration

```python
from hqpf.config import Config

# Load config
config = Config.from_yaml('config/default.yaml')

# Create model from config
model = HybridModel.from_config(config)

# Update config
config.model.n_gnn_layers = 6
config.training.batch_size = 64
config.save('config/custom.yaml')
```

## Utilities

### Sequence Processing

```python
from hqpf.utils import (
    sequence_to_tensor,
    tensor_to_sequence,
    one_hot_encode,
    compute_sequence_features
)

# Convert sequence
seq_str = "ACDEFGHIKL"
seq_tensor = sequence_to_tensor(seq_str, device='cuda')

# One-hot encoding
onehot = one_hot_encode(seq_tensor)  # (n_residues, 20)

# Features
features = compute_sequence_features(seq_str)
# Returns: {'length': 10, 'hydrophobicity': 0.42, ...}
```

### Structure Processing

```python
from hqpf.utils import (
    center_structure,
    align_structures,
    compute_distances,
    compute_angles
)

# Center at origin
centered = center_structure(structure)

# Align to reference
aligned = align_structures(predicted, native)

# Distance matrix
dist_matrix = compute_distances(structure)  # (N, N)

# Backbone angles
angles = compute_angles(structure)  # phi, psi, omega
```

## Command Line Interface

```bash
# Predict structure
hqpf predict --sequence ACDEFG --output structure.pdb

# Train model
hqpf train --config config/train.yaml --data data/

# Evaluate
hqpf evaluate --model checkpoints/best.pt --test-set data/test/

# Benchmark
hqpf benchmark --dataset CASP14 --metrics all
```

## Example Workflows

### Single Prediction

```python
import torch
from hqpf.models import HybridModel
from hqpf.utils import sequence_to_tensor

# Setup
sequence = "ACDEFGHIKL"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize
model = HybridModel(
    n_residues=len(sequence),
    n_qubits=len(sequence),
    use_quantum=True,
    device=device
)

# Predict
seq_tensor = sequence_to_tensor(sequence, device=device)
model.eval()
with torch.no_grad():
    outputs = model(seq_tensor, n_candidates=10)

# Save
structure = outputs['best_structure'].cpu().numpy()
np.save('structure.npy', structure)
```

### Batch Prediction

```python
from hqpf.data import ProteinDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = ProteinDataset('proteins.fasta')
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Predict all
results = []
for batch in loader:
    with torch.no_grad():
        outputs = model(batch)
    results.extend(outputs['best_structure'])
```

## Advanced Features

### Custom Loss Functions

```python
class CustomLoss(nn.Module):
    def forward(self, predicted, target):
        # Your loss computation
        return loss

model.loss_fn = CustomLoss()
```

### Callbacks

```python
from hqpf.training import Callback

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        print(f"Epoch {epoch}: {logs}")

trainer.add_callback(MyCallback())
```

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model)

# Train normally
trainer.train(epochs=100)
```
