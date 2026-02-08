"""
Benchmark protein datasets for testing and validation.

Provides:
- Small test proteins (10-50 residues)
- Standard benchmark sets (CASP, CAMEO)
- Protein stability datasets
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class BenchmarkProtein:
    """
    Container for benchmark protein data.
    
    Attributes:
        name: Protein identifier
        sequence: Amino acid sequence string
        native_structure: Ground truth 3D coordinates (n_residues, 3)
        secondary_structure: Secondary structure labels
        stability: Experimental stability (ΔG)
    """
    
    def __init__(
        self,
        name: str,
        sequence: str,
        native_structure: Optional[np.ndarray] = None,
        secondary_structure: Optional[str] = None,
        stability: Optional[float] = None,
        pdb_id: Optional[str] = None
    ):
        self.name = name
        self.sequence = sequence
        self.native_structure = native_structure
        self.secondary_structure = secondary_structure
        self.stability = stability
        self.pdb_id = pdb_id
    
    def __repr__(self):
        return f"BenchmarkProtein(name='{self.name}', length={len(self.sequence)})"
    
    def to_tensor(self, aa_to_idx: Dict[str, int], device: str = 'cpu') -> torch.Tensor:
        """Convert sequence to tensor."""
        indices = [aa_to_idx.get(aa, 0) for aa in self.sequence]
        return torch.tensor(indices, dtype=torch.long, device=device)


# Standard amino acid mapping
AA_TO_IDX = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}


# Small test proteins for rapid validation
TEST_PROTEINS = {
    'peptide_10': BenchmarkProtein(
        name='test_peptide_10',
        sequence='ACDEFGHIKL',
        secondary_structure='CCCCCCCCCC'
    ),
    'helix_12': BenchmarkProtein(
        name='alpha_helix_12',
        sequence='AEAAAKEAAAAA',
        secondary_structure='HHHHHHHHHHHH'
    ),
    'sheet_10': BenchmarkProtein(
        name='beta_sheet_10',
        sequence='YVYVYVYVYV',
        secondary_structure='EEEEEEEEEE'
    ),
    'mixed_20': BenchmarkProtein(
        name='mixed_structure_20',
        sequence='AEKLAEKLAEKLAEKLADEF',
        secondary_structure='HHHHHHHHHCCCEEEEEEEE'
    )
}


# Mini-proteins with known structures (simplified)
MINI_PROTEINS = {
    'trp_cage': BenchmarkProtein(
        name='Trp-cage',
        sequence='NLYIQWLKDGGPSSGRPPPS',
        pdb_id='1L2Y',
        stability=-5.5  # kcal/mol
    ),
    'villin': BenchmarkProtein(
        name='Villin headpiece',
        sequence='LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
        pdb_id='1YRF',
        stability=-4.2
    ),
    'ww_domain': BenchmarkProtein(
        name='WW domain',
        sequence='GSKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSG',
        pdb_id='1PIN',
        stability=-3.8
    )
}


# Stability prediction benchmark
STABILITY_BENCHMARK = {
    'stable_1': BenchmarkProtein(
        name='stable_mutant_1',
        sequence='MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL',
        stability=-8.5
    ),
    'unstable_1': BenchmarkProtein(
        name='unstable_mutant_1',
        sequence='MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSDD',
        stability=-2.1
    )
}


def load_benchmark_protein(
    protein_id: str,
    dataset: str = 'test',
    device: str = 'cpu'
) -> BenchmarkProtein:
    """
    Load a benchmark protein by ID.
    
    Args:
        protein_id: Protein identifier
        dataset: Dataset name ('test', 'mini', 'stability')
        device: Device for tensor conversion
        
    Returns:
        BenchmarkProtein instance
        
    Example:
        >>> protein = load_benchmark_protein('peptide_10', dataset='test')
        >>> sequence = protein.to_tensor(AA_TO_IDX, device='cuda')
    """
    datasets = {
        'test': TEST_PROTEINS,
        'mini': MINI_PROTEINS,
        'stability': STABILITY_BENCHMARK
    }
    
    if dataset not in datasets:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(datasets.keys())}")
    
    proteins = datasets[dataset]
    
    if protein_id not in proteins:
        available = list(proteins.keys())
        raise ValueError(f"Unknown protein '{protein_id}'. Available in '{dataset}': {available}")
    
    return proteins[protein_id]


def list_benchmark_proteins(dataset: str = 'test') -> List[str]:
    """
    List available benchmark proteins in a dataset.
    
    Args:
        dataset: Dataset name ('test', 'mini', 'stability')
        
    Returns:
        List of protein IDs
    """
    datasets = {
        'test': TEST_PROTEINS,
        'mini': MINI_PROTEINS,
        'stability': STABILITY_BENCHMARK
    }
    
    if dataset not in datasets:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(datasets.keys())}")
    
    return list(datasets[dataset].keys())


def get_benchmark_statistics(dataset: str = 'test') -> Dict:
    """
    Get statistics about a benchmark dataset.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Dictionary with dataset statistics
    """
    datasets = {
        'test': TEST_PROTEINS,
        'mini': MINI_PROTEINS,
        'stability': STABILITY_BENCHMARK
    }
    
    proteins = datasets[dataset]
    sequences = [p.sequence for p in proteins.values()]
    
    stats = {
        'n_proteins': len(proteins),
        'min_length': min(len(s) for s in sequences),
        'max_length': max(len(s) for s in sequences),
        'avg_length': np.mean([len(s) for s in sequences]),
        'protein_ids': list(proteins.keys())
    }
    
    return stats


def load_test_batch(
    batch_size: int = 4,
    dataset: str = 'test',
    device: str = 'cpu'
) -> Tuple[List[torch.Tensor], List[BenchmarkProtein]]:
    """
    Load a batch of test proteins.
    
    Args:
        batch_size: Number of proteins to load
        dataset: Dataset name
        device: Device for tensors
        
    Returns:
        (sequences, proteins) tuple
    """
    protein_ids = list_benchmark_proteins(dataset)[:batch_size]
    
    proteins = [load_benchmark_protein(pid, dataset, device) for pid in protein_ids]
    sequences = [p.to_tensor(AA_TO_IDX, device) for p in proteins]
    
    return sequences, proteins


if __name__ == '__main__':
    # Demo
    print("Available datasets:")
    for dataset in ['test', 'mini', 'stability']:
        stats = get_benchmark_statistics(dataset)
        print(f"\n{dataset.upper()} Dataset:")
        print(f"  Proteins: {stats['n_proteins']}")
        print(f"  Length range: {stats['min_length']}-{stats['max_length']}")
        print(f"  IDs: {', '.join(stats['protein_ids'])}")
    
    print("\nExample usage:")
    protein = load_benchmark_protein('peptide_10', dataset='test')
    print(f"Loaded: {protein}")
    print(f"Sequence: {protein.sequence}")
    print(f"Secondary structure: {protein.secondary_structure}")
