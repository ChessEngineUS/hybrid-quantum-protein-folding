"""
Hybrid Quantum-AI Protein Folding (HQPF)

A framework for protein structure prediction using VQE + deep learning.
"""

__version__ = "0.1.0"
__author__ = "Tommaso R. Marena"

from hqpf.models.hybrid_model import HybridQuantumModel
from hqpf.data.benchmarks import load_benchmark_protein
from hqpf.training.trainer import Trainer

__all__ = [
    "HybridQuantumModel",
    "load_benchmark_protein",
    "Trainer",
]
