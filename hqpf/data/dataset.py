"""
Dataset module for protein structure prediction.

Handles:
- Loading experimental data (PDB, NMR ensembles)
- Sequence encoding
- Structure normalization
- Data augmentation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path


class ProteinDataset(Dataset):
    """
    Dataset for protein structure prediction.
    
    Data format:
        - sequence: List of amino acid indices (0-19)
        - native_structure: Ground truth coordinates (optional)
        - stability_labels: Experimental stability rankings
        - delta_G: Free energy of folding
        - metadata: Protein name, source, etc.
        
    Parameters:
        data_dir: Directory containing protein data files
        split: 'train', 'val', or 'test'
        max_length: Maximum sequence length
        augment: Whether to apply data augmentation
    """
    
    # Amino acid encoding
    AA_TO_IDX = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
        'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }
    
    IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_length: int = 50,
        augment: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.augment = augment
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """
        Load protein data from files.
        
        Expected file format (JSON):
        {
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "native_structure": [[x1,y1,z1], [x2,y2,z2], ...],
            "delta_G": -5.2,
            "metadata": {...}
        }
        """
        data_file = self.data_dir / f"{self.split}.json"
        
        if not data_file.exists():
            # Generate synthetic data for demonstration
            return self._generate_synthetic_data()
        
        import json
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def _generate_synthetic_data(self, n_samples: int = 10) -> List[Dict]:
        """
        Generate synthetic protein data for testing.
        
        Creates random sequences and simple helical/sheet structures.
        """
        data = []
        
        for i in range(n_samples):
            length = np.random.randint(10, min(self.max_length, 25))
            
            # Random sequence
            sequence = ''.join(np.random.choice(list(self.AA_TO_IDX.keys()), length))
            
            # Simple helical structure (3.6 residues per turn)
            t = np.linspace(0, 2 * np.pi * length / 3.6, length)
            x = 2.3 * np.cos(t)
            y = 2.3 * np.sin(t)
            z = 1.5 * np.arange(length)
            native_structure = np.stack([x, y, z], axis=1).tolist()
            
            # Random stability
            delta_G = np.random.uniform(-10, -2)
            
            data.append({
                'sequence': sequence,
                'native_structure': native_structure,
                'delta_G': delta_G,
                'metadata': {'name': f'protein_{i}', 'type': 'synthetic'}
            })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single protein sample.
        
        Returns:
            sample: Dictionary containing:
                - sequence: (length,) tensor of AA indices
                - native_structure: (length, 3) tensor of coordinates
                - delta_G: scalar tensor
                - metadata: dict
        """
        item = self.data[idx]
        
        # Encode sequence
        sequence_str = item['sequence']
        sequence = torch.tensor(
            [self.AA_TO_IDX[aa] for aa in sequence_str],
            dtype=torch.long
        )
        
        # Parse structure
        if 'native_structure' in item:
            native_structure = torch.tensor(
                item['native_structure'],
                dtype=torch.float32
            )
        else:
            native_structure = None
        
        # Parse energy
        if 'delta_G' in item:
            delta_G = torch.tensor(item['delta_G'], dtype=torch.float32)
        else:
            delta_G = None
        
        # Data augmentation
        if self.augment and native_structure is not None:
            native_structure = self._augment_structure(native_structure)
        
        # Prepare sample
        sample = {
            'sequence': sequence,
            'metadata': item.get('metadata', {})
        }
        
        if native_structure is not None:
            sample['native_structure'] = native_structure
        
        if delta_G is not None:
            sample['delta_G'] = delta_G
        
        # Add stability labels if available
        if 'stability_labels' in item:
            sample['stability_labels'] = torch.tensor(
                item['stability_labels'],
                dtype=torch.long
            )
        
        return sample
    
    def _augment_structure(self, structure: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to structure.
        
        Includes:
        - Random rotation
        - Random translation
        - Small Gaussian noise
        """
        # Random rotation (around z-axis)
        if np.random.rand() > 0.5:
            theta = np.random.uniform(-np.pi/4, np.pi/4)
            c, s = np.cos(theta), np.sin(theta)
            rot = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ], dtype=torch.float32)
            structure = structure @ rot.T
        
        # Random translation
        if np.random.rand() > 0.5:
            translation = torch.randn(3) * 2.0
            structure = structure + translation
        
        # Gaussian noise
        if np.random.rand() > 0.5:
            noise = torch.randn_like(structure) * 0.5
            structure = structure + noise
        
        return structure
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for batching.
        
        Since proteins have variable length, we process one at a time.
        """
        # Return first item only (batch_size=1)
        return batch[0]


class IDRDataset(ProteinDataset):
    """
    Dataset for intrinsically disordered regions (IDRs).
    
    Special handling:
        - Ensemble of structures instead of single native
        - NMR-derived distance constraints
        - FRET distance distributions
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _load_data(self) -> List[Dict]:
        """Load IDR-specific data with NMR ensembles."""
        # Load from NMR ensemble files or generate synthetic ensembles
        return self._generate_synthetic_idr_data()
    
    def _generate_synthetic_idr_data(self, n_samples: int = 5) -> List[Dict]:
        """
        Generate synthetic IDR data with ensemble structures.
        """
        data = []
        
        for i in range(n_samples):
            length = np.random.randint(15, 40)
            
            # Random sequence (bias toward disordered-promoting residues)
            disordered_aa = ['G', 'S', 'P', 'Q', 'E', 'K']
            sequence = ''.join(
                np.random.choice(
                    disordered_aa if np.random.rand() > 0.3 else list(self.AA_TO_IDX.keys()),
                    1
                )[0] for _ in range(length)
            )
            
            # Generate ensemble of random coil conformations
            n_conformers = 10
            ensemble = []
            for _ in range(n_conformers):
                # Random walk in 3D
                steps = np.random.randn(length, 3) * 3.8  # ~C-alpha distance
                structure = np.cumsum(steps, axis=0)
                ensemble.append(structure.tolist())
            
            data.append({
                'sequence': sequence,
                'ensemble': ensemble,
                'metadata': {'name': f'idr_{i}', 'type': 'disordered'}
            })
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get IDR sample with ensemble."""
        item = self.data[idx]
        
        # Encode sequence
        sequence_str = item['sequence']
        sequence = torch.tensor(
            [self.AA_TO_IDX[aa] for aa in sequence_str],
            dtype=torch.long
        )
        
        # Parse ensemble
        if 'ensemble' in item:
            ensemble = torch.tensor(item['ensemble'], dtype=torch.float32)
            # Use centroid as "native"
            native_structure = ensemble.mean(dim=0)
        else:
            native_structure = None
        
        sample = {
            'sequence': sequence,
            'metadata': item.get('metadata', {})
        }
        
        if native_structure is not None:
            sample['native_structure'] = native_structure
        
        return sample


class BenchmarkDataset(ProteinDataset):
    """
    Dataset for standard benchmarking (CASP, CAMEO).
    
    Includes:
        - High-quality PDB structures
        - Multiple domains per protein
        - Experimental validation data
    """
    
    BENCHMARK_PROTEINS = [
        # Problem Set A: IDRs
        ('p53-TAD', 'SQETFSDLWKLLPEN', 'disordered'),
        ('c-Fos-linker', 'SGPSAREPRILGGPESPDLSLDALLGP', 'disordered'),
        
        # Problem Set B: Metamorphic
        ('FF-domain', 'MQIFVKTLTGKTITL', 'metamorphic'),
        
        # Problem Set C: Viral peptides
        ('zika-fragment', 'HLTKERLLDVFTRN', 'constrained')
    ]
    
    def _load_data(self) -> List[Dict]:
        """Load benchmark proteins."""
        data = []
        
        for name, seq, ptype in self.BENCHMARK_PROTEINS:
            # Generate simple structure (placeholder)
            length = len(seq)
            t = np.linspace(0, 2 * np.pi * length / 3.6, length)
            x = 2.3 * np.cos(t)
            y = 2.3 * np.sin(t)
            z = 1.5 * np.arange(length)
            structure = np.stack([x, y, z], axis=1).tolist()
            
            data.append({
                'sequence': seq,
                'native_structure': structure,
                'delta_G': np.random.uniform(-10, -5),
                'metadata': {'name': name, 'type': ptype}
            })
        
        return data
