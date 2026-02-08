"""Performance benchmark tests."""

import pytest
import torch
from hqpf.models.hybrid_model import HybridModel


class TestPerformance:
    """Benchmark model performance."""
    
    def test_forward_pass_speed(self, benchmark):
        """Benchmark forward pass speed."""
        model = HybridModel(n_residues=10, n_qubits=10, use_quantum=False, device='cpu')
        sequence = torch.randint(0, 20, (10,), dtype=torch.long)
        model.eval()
        
        def forward():
            with torch.no_grad():
                return model(sequence, n_candidates=5)
        
        result = benchmark(forward)
        assert result is not None
    
    def test_ensemble_generation_speed(self, benchmark):
        """Benchmark ensemble generation."""
        from hqpf.models.structure_generator import StructureGenerator
        
        generator = StructureGenerator(n_residues=10, device='cpu')
        sequence = torch.randint(0, 20, (10,), dtype=torch.long)
        
        def generate_ensemble():
            return generator.generate_ensemble(sequence, n_samples=10)
        
        result = benchmark(generate_ensemble)
        assert len(result[0]) == 10
    
    @pytest.mark.parametrize('n_residues', [10, 20, 30])
    def test_scalability(self, benchmark, n_residues):
        """Test performance scaling with protein size."""
        model = HybridModel(n_residues=n_residues, n_qubits=n_residues, use_quantum=False, device='cpu')
        sequence = torch.randint(0, 20, (n_residues,), dtype=torch.long)
        model.eval()
        
        def forward():
            with torch.no_grad():
                return model(sequence, n_candidates=3)
        
        result = benchmark(forward)
        assert result['best_structure'].shape == (n_residues, 3)
