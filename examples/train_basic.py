#!/usr/bin/env python3
"""
Basic training example for hybrid quantum-classical protein folding.

Usage:
    python examples/train_basic.py --config config/basic.json
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
import json
from pathlib import Path

from hqpf.models.hybrid_model import HybridModel
from hqpf.data import ProteinDataset, BenchmarkDataset
from hqpf.training import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train hybrid quantum-classical protein folding model')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--use_quantum', action='store_true', help='Use real quantum hardware')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command-line args
    config.update({
        'checkpoint_dir': args.checkpoint_dir,
        'n_epochs': args.n_epochs,
        'device': args.device
    })
    
    print("=" * 80)
    print("Hybrid Quantum-Classical Protein Folding - Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Using quantum hardware: {args.use_quantum}")
    print(f"Data directory: {args.data_dir}")
    print()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = ProteinDataset(
        data_dir=args.data_dir,
        split='train',
        max_length=25,
        augment=True
    )
    
    val_dataset = ProteinDataset(
        data_dir=args.data_dir,
        split='val',
        max_length=25,
        augment=False
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print()
    
    # Create model
    print("Initializing model...")
    model = HybridModel(
        n_residues=25,
        n_qubits=20,
        embedding_dim=128,
        hidden_dim=256,
        n_gnn_layers=4,
        n_vqe_layers=3,
        use_quantum=args.use_quantum,
        backend='qiskit',
        output_type='lattice',
        device=args.device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create trainer
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train
    print("Starting training...")
    print("=" * 80)
    metrics_history = trainer.train()
    
    print()
    print("=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final train loss: {metrics_history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {metrics_history['val_loss'][-1]:.4f}")
    print()
    
    # Benchmark on test set
    print("Running benchmark...")
    test_dataset = BenchmarkDataset(
        data_dir=args.data_dir,
        split='test'
    )
    
    results = trainer.benchmark(test_dataset)
    
    print("Benchmark results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    print()
    print("Done!")


if __name__ == '__main__':
    main()
