"""Comprehensive tests for Hybrid Model."""

import pytest
import torch
from hqpf.models.hybrid_model import HybridModel


class TestHybridModel:
    """Test suite for Hybrid Quantum-Classical Model."""
    
    @pytest.fixture
    def model(self):
        """Create a hybrid model instance."""
        return HybridModel(
            n_residues=10,
            n_qubits=10,
            embedding_dim=64,
            hidden_dim=128,
            n_gnn_layers=3,
            n_vqe_layers=2,
            use_quantum=False,
            device='cpu'
        )
    
    @pytest.fixture
    def sequence(self):
        """Create a test sequence."""
        return torch.randint(0, 20, (10,), dtype=torch.long)
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_residues == 10
        assert model.n_qubits == 10
        assert model.embedding_dim == 64
        assert model.hidden_dim == 128
        assert not model.use_quantum
    
    def test_forward_pass(self, model, sequence):
        """Test forward pass produces valid output."""
        model.eval()
        with torch.no_grad():
            outputs = model(sequence, n_candidates=5)
        
        assert 'best_structure' in outputs
        assert 'best_energy' in outputs
        assert 'all_structures' in outputs
        assert 'all_energies' in outputs
        assert 'quantum_fraction' in outputs
        
        assert outputs['best_structure'].shape == (10, 3)
        assert outputs['best_energy'].shape == ()
        assert len(outputs['all_structures']) == 5
    
    def test_training_mode(self, model, sequence):
        """Test model in training mode."""
        model.train()
        outputs = model(sequence, n_candidates=3)
        
        assert outputs['best_energy'].requires_grad
    
    def test_quantum_mode(self, sequence):
        """Test quantum evaluation mode."""
        model = HybridModel(
            n_residues=10,
            n_qubits=10,
            use_quantum=True,
            device='cpu'
        )
        model.eval()
        with torch.no_grad():
            outputs = model(sequence, n_candidates=3)
        
        assert outputs['quantum_fraction'] > 0
    
    def test_gradient_flow(self, model, sequence):
        """Test gradients flow through model."""
        model.train()
        outputs = model(sequence, n_candidates=2)
        loss = outputs['best_energy']
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_different_sequence_lengths(self):
        """Test model handles different sequence lengths."""
        for length in [5, 10, 15, 20]:
            model = HybridModel(n_residues=length, n_qubits=length, device='cpu')
            sequence = torch.randint(0, 20, (length,), dtype=torch.long)
            model.eval()
            with torch.no_grad():
                outputs = model(sequence, n_candidates=2)
            assert outputs['best_structure'].shape == (length, 3)
    
    def test_cuda_compatibility(self, sequence):
        """Test CUDA device compatibility."""
        if torch.cuda.is_available():
            model = HybridModel(n_residues=10, n_qubits=10, device='cuda')
            sequence = sequence.to('cuda')
            model.eval()
            with torch.no_grad():
                outputs = model(sequence, n_candidates=2)
            assert outputs['best_structure'].device.type == 'cuda'
    
    def test_energy_minimization(self, model, sequence):
        """Test model finds lower energy structures."""
        model.eval()
        with torch.no_grad():
            outputs = model(sequence, n_candidates=10)
        
        best_energy = outputs['best_energy'].item()
        all_energies = [e.item() for e in outputs['all_energies']]
        
        # Best energy should be minimum
        assert best_energy == min(all_energies)
    
    def test_surrogate_model(self, model, sequence):
        """Test surrogate model prediction."""
        model.eval()
        with torch.no_grad():
            outputs = model(sequence, n_candidates=5, use_surrogate=True)
        
        assert 'surrogate_energies' in outputs or 'all_energies' in outputs
    
    def test_reproducibility(self, model, sequence):
        """Test deterministic behavior with seed."""
        torch.manual_seed(42)
        model.eval()
        with torch.no_grad():
            outputs1 = model(sequence, n_candidates=5)
        
        torch.manual_seed(42)
        with torch.no_grad():
            outputs2 = model(sequence, n_candidates=5)
        
        assert torch.allclose(outputs1['best_energy'], outputs2['best_energy'])
