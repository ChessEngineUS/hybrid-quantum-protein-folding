# Google Colab Quick Start Instructions

## ⚡ Quick Fix for Import Errors

If you encounter ANY errors during setup, follow these steps:

### Step 1: Restart Runtime
1. Go to **Runtime** menu → **Restart runtime**
2. Click **Yes** to confirm

### Step 2: Run This Setup Cell

After restarting, run this single cell:

```python
# === COMPLETE SETUP CELL - Run after restart ===
import sys
import os
import importlib

# Navigate to repository
if not os.path.exists('/content/hybrid-quantum-protein-folding'):
    print("❌ Repository not found! Please run the installation cell first.")
else:
    os.chdir('/content/hybrid-quantum-protein-folding')
    
    # Clear ALL cached modules
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('hqpf')]
    for module in modules_to_remove:
        del sys.modules[module]
    
    # Pull latest changes
    !git pull origin main --quiet
    
    # Now import
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from hqpf.models import HybridModel
    
    print("✅ Setup complete!")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
```

### Step 3: Run Prediction

```python
# Define sequence
sequence_str = "ACDEFGHIKL"
aa_to_idx = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}
sequence = torch.tensor([aa_to_idx[aa] for aa in sequence_str], dtype=torch.long)

print(f'Sequence: {sequence_str}')
print(f'Length: {len(sequence)} residues')

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\nUsing device: {device}')

model = HybridModel(
    n_residues=len(sequence),
    n_qubits=10,
    use_quantum=False,
    device=device
)

print('✅ Model initialized!')

# Predict
model.eval()
with torch.no_grad():
    outputs = model(sequence, n_candidates=10)

best_structure = outputs['best_structure']
best_energy = outputs['best_energy']

print(f'\n✅ Prediction complete!')
print(f'Best energy: {best_energy.item():.4f}')
print(f'Structure shape: {best_structure.shape}')
```

---

## 🐛 Common Errors and Solutions

### Error: `KeyError: "attribute 'parameters' already exists"`
**Solution:** This was a naming conflict with PyTorch. Fixed in commit `2ee90e2`. Run `git pull origin main` to update.

### Error: `TypeError: VQESolver.__init__() got an unexpected keyword argument 'n_layers'`
**Solution:** Parameter name changed to `ansatz_depth`. Fixed in commit `e1bd443`. Run `git pull origin main` to update.

### Error: `cannot import name 'HybridQuantumModel'`
**Solution:** Class renamed to `HybridModel` with alias. Fixed in commit `bfe3a0c`. Import as: `from hqpf.models import HybridModel`

### Error: `ImportError: cannot import name 'Estimator'`
**Solution:** Qiskit 1.0+ compatibility added. Fixed in commit `7b27e74`. Module has fallback imports.

### Error: `ModuleNotFoundError: No module named 'hqpf'`
**Solution:** 
1. Make sure you ran the installation cell
2. Check you're in the correct directory: `%cd /content/hybrid-quantum-protein-folding`
3. Restart runtime and skip installation cell

### Warning: `Could not import load_benchmark_protein`
**Status:** Non-critical warning. Benchmark loading is optional. Model works fine.

### Warning: `pip's dependency resolver` conflicts
**Status:** Non-critical. Google Colab's ipython/notebook versions conflict with requirements but don't affect functionality.

---

## 🚀 Performance Tips

### GPU Acceleration
- **Recommended:** T4 GPU (free tier)
- **Best:** A100 GPU (Colab Pro)
- **Change runtime:** Runtime → Change runtime type → Hardware accelerator → T4 GPU

### Memory Management
```python
# If you get CUDA out of memory:
model = HybridModel(
    n_residues=len(sequence),
    n_qubits=10,
    use_quantum=False,
    device='cpu'  # Force CPU
)

# Or reduce candidates:
outputs = model(sequence, n_candidates=5)  # Instead of 10
```

### Faster Predictions
```python
# Use surrogate model (much faster than quantum simulation)
model.eval()
with torch.no_grad():
    outputs = model(
        sequence, 
        n_candidates=10,
        use_surrogate=True  # 10-100x faster
    )
```

---

## 📖 Complete Working Example

Here's a complete, copy-paste ready example that works:

```python
# === AFTER INSTALLATION AND RESTART ===

import sys
import os
os.chdir('/content/hybrid-quantum-protein-folding')

# Clear cache
for k in list(sys.modules.keys()):
    if k.startswith('hqpf'):
        del sys.modules[k]

# Import
import torch
import numpy as np
from hqpf.models import HybridModel

# Sequence
sequence_str = "ACDEFGHIKL"
aa_to_idx = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}
sequence = torch.tensor([aa_to_idx[aa] for aa in sequence_str], dtype=torch.long)

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HybridModel(n_residues=len(sequence), n_qubits=10, use_quantum=False, device=device)

# Predict
model.eval()
with torch.no_grad():
    outputs = model(sequence, n_candidates=10)

print(f"✅ Success! Energy: {outputs['best_energy'].item():.4f}")
print(f"Structure shape: {outputs['best_structure'].shape}")
```

---

## 📞 Support

- **GitHub Issues:** [Report bugs](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/issues)
- **Documentation:** [Read the docs](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/tree/main/docs)
- **Examples:** [See examples](https://github.com/ChessEngineUS/hybrid-quantum-protein-folding/tree/main/examples)

## ✅ Verification Checklist

Before reporting issues, verify:

- [ ] Ran installation cell successfully
- [ ] Restarted runtime (Runtime → Restart runtime)
- [ ] Ran `git pull origin main` to get latest fixes
- [ ] Cleared module cache with the code above
- [ ] In correct directory (`/content/hybrid-quantum-protein-folding`)
- [ ] Using Python 3.10+
- [ ] PyTorch 2.0+ installed
- [ ] CUDA available (if using GPU)

---

**Last Updated:** Feb 8, 2026  
**Latest Stable Commit:** `2ee90e2`  
**Status:** ✅ All critical bugs fixed
