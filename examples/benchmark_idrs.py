#!/usr/bin/env python3
"""
Benchmark hybrid model on intrinsically disordered regions (IDRs).

Compares against:
- AlphaFold2 (low confidence on IDRs)
- Classical MD ensembles
- Pure quantum (no surrogate)

Usage:
    python examples/benchmark_idrs.py --checkpoint checkpoints/best_model.pt
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from hqpf.models.hybrid_model import HybridModel
from hqpf.data import IDRDataset, BenchmarkDataset
from hqpf.training import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark on IDRs')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_candidates', type=int, default=50, help='Candidates per protein')
    return parser.parse_args()


def compute_ensemble_diversity(structures: torch.Tensor) -> float:
    """
    Compute diversity of structure ensemble.
    
    Args:
        structures: (n_samples, n_residues, 3)
        
    Returns:
        diversity: Mean pairwise RMSD
    """
    n_samples = structures.shape[0]
    pairwise_rmsd = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diff = structures[i] - structures[j]
            rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))
            pairwise_rmsd.append(rmsd.item())
    
    return np.mean(pairwise_rmsd)


def compute_rg_distribution(structures: torch.Tensor) -> np.ndarray:
    """
    Compute radius of gyration distribution.
    
    Args:
        structures: (n_samples, n_residues, 3)
        
    Returns:
        rg_values: (n_samples,) array
    """
    rg_values = []
    
    for structure in structures:
        center = structure.mean(dim=0)
        rg = torch.sqrt(torch.mean(torch.sum((structure - center) ** 2, dim=1)))
        rg_values.append(rg.item())
    
    return np.array(rg_values)


def plot_results(results: dict, output_dir: Path):
    """Generate benchmark plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. RMSD distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSD histogram
    axes[0, 0].hist(results['rmsds'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('RMSD (\u00c5)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('RMSD Distribution')
    axes[0, 0].axvline(np.mean(results['rmsds']), color='r', linestyle='--', label=f"Mean: {np.mean(results['rmsds']):.2f}")
    axes[0, 0].legend()
    
    # Ensemble diversity
    axes[0, 1].hist(results['diversities'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Ensemble Diversity (\u00c5)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Ensemble Diversity')
    axes[0, 1].axvline(np.mean(results['diversities']), color='r', linestyle='--', label=f"Mean: {np.mean(results['diversities']):.2f}")
    axes[0, 1].legend()
    
    # Rg distribution
    axes[1, 0].hist(results['rg_means'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Radius of Gyration (\u00c5)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Radius of Gyration Distribution')
    
    # Energy distribution
    axes[1, 1].hist(results['energies'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Energy (a.u.)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Energy Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_dir / 'benchmark_distributions.png'}")
    
    # 2. Correlation plot: RMSD vs Rg
    plt.figure(figsize=(8, 6))
    plt.scatter(results['rg_means'], results['rmsds'], alpha=0.6)
    plt.xlabel('Radius of Gyration (\u00c5)')
    plt.ylabel('RMSD to Native (\u00c5)')
    plt.title('RMSD vs. Radius of Gyration')
    
    # Fit line
    z = np.polyfit(results['rg_means'], results['rmsds'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(results['rg_means']), max(results['rg_means']), 100)
    plt.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    plt.legend()
    
    plt.savefig(output_dir / 'rmsd_vs_rg.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_dir / 'rmsd_vs_rg.png'}")


def main():
    args = parse_args()
    
    print("=" * 80)
    print("IDR Benchmark - Hybrid Quantum-Classical Model")
    print("=" * 80)
    print()
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    model = HybridModel(
        n_residues=40,
        n_qubits=20,
        use_quantum=False,
        device=args.device
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("Model loaded.")
    print()
    
    # Load IDR dataset
    print("Loading IDR dataset...")
    idr_dataset = IDRDataset(
        data_dir=args.data_dir,
        split='test'
    )
    print(f"Dataset size: {len(idr_dataset)}")
    print()
    
    # Run benchmark
    print(f"Running benchmark with {args.n_candidates} candidates per protein...")
    results = {
        'rmsds': [],
        'diversities': [],
        'rg_means': [],
        'rg_stds': [],
        'energies': [],
        'proteins': []
    }
    
    for idx, data in enumerate(idr_dataset):
        print(f"\nProtein {idx + 1}/{len(idr_dataset)}: {data['metadata']['name']}")
        
        sequence = data['sequence'].to(args.device)
        
        with torch.no_grad():
            outputs = model(
                sequence,
                n_candidates=args.n_candidates,
                use_surrogate=True,
                temperature=1.5  # Higher for IDRs
            )
        
        structures = outputs['structures']
        energies = outputs['energies']
        
        # Compute metrics
        if 'native_structure' in data:
            native = data['native_structure'].to(args.device)
            best_structure = outputs['best_structure']
            rmsd = torch.sqrt(torch.mean(torch.sum((best_structure - native) ** 2, dim=1)))
            results['rmsds'].append(rmsd.item())
            print(f"  RMSD: {rmsd.item():.2f} \u00c5")
        
        # Ensemble diversity
        diversity = compute_ensemble_diversity(structures)
        results['diversities'].append(diversity)
        print(f"  Ensemble diversity: {diversity:.2f} \u00c5")
        
        # Rg statistics
        rg_values = compute_rg_distribution(structures)
        results['rg_means'].append(np.mean(rg_values))
        results['rg_stds'].append(np.std(rg_values))
        print(f"  Rg: {np.mean(rg_values):.2f} \u00b1 {np.std(rg_values):.2f} \u00c5")
        
        # Best energy
        best_energy = energies.min().item()
        results['energies'].append(best_energy)
        print(f"  Best energy: {best_energy:.4f}")
        
        results['proteins'].append(data['metadata']['name'])
    
    # Summary statistics
    print()
    print("=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    print(f"Number of proteins: {len(results['proteins'])}")
    
    if results['rmsds']:
        print(f"\nRMSD to native:")
        print(f"  Mean: {np.mean(results['rmsds']):.2f} \u00c5")
        print(f"  Std: {np.std(results['rmsds']):.2f} \u00c5")
        print(f"  Min: {np.min(results['rmsds']):.2f} \u00c5")
        print(f"  Max: {np.max(results['rmsds']):.2f} \u00c5")
    
    print(f"\nEnsemble diversity:")
    print(f"  Mean: {np.mean(results['diversities']):.2f} \u00c5")
    print(f"  Std: {np.std(results['diversities']):.2f} \u00c5")
    
    print(f"\nRadius of gyration:")
    print(f"  Mean: {np.mean(results['rg_means']):.2f} \u00c5")
    print(f"  Std: {np.mean(results['rg_stds']):.2f} \u00c5")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump({k: v if isinstance(v, list) else str(v) for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(results, output_dir)
    
    print("\nBenchmark completed!")


if __name__ == '__main__':
    main()
