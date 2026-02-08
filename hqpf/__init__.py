"""
Hybrid Quantum-AI Protein Folding (HQPF)

A framework for protein structure prediction using VQE + deep learning.
"""

__version__ = "0.1.0"
__author__ = "Tommaso R. Marena"

# Try to import main components, with graceful fallback
try:
    from hqpf.models.hybrid_model import HybridQuantumModel
    # Expose as both names for convenience
    HybridModel = HybridQuantumModel
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import HybridQuantumModel: {e}")
    HybridQuantumModel = None
    HybridModel = None

try:
    from hqpf.data.benchmarks import load_benchmark_protein
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import load_benchmark_protein: {e}")
    load_benchmark_protein = None

try:
    from hqpf.training.trainer import Trainer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import Trainer: {e}")
    Trainer = None

__all__ = [
    "HybridQuantumModel",
    "HybridModel",  # Alias for convenience
    "load_benchmark_protein",
    "Trainer",
]
