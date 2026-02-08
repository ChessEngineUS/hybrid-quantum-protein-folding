"""Tests for Structure Generator."""

import pytest
import torch
from hqpf.models.structure_generator import StructureGenerator


class TestStructureGenerator:
    """Test suite for structure generation."""
    
    @pytest.fixture
    def generator(self):
        """Create a structure generator."""
        return StructureGenerator(
            n_residues=10,
            embedding_dim=64,
            hidden_dim=128,
            n_layers=3,
            device='cpu'
        )
    
    @pytest.fixture
    def sequence(self):
        """Create a test sequence."""
        return torch.randint(0, 20, (10,), dtype=torch.long)
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.n_residues == 10
        assert generator.embedding_dim == 64
        assert generator.hidden_dim == 128
    
    def test_forward_pass(self, generator, sequence):
        """Test forward pass produces 3D coordinates."""
        structure, prior = generator(sequence)
        
        assert structure.shape == (10, 3)
        assert torch.all(torch.isfinite(structure))
        if prior is not None:
            assert prior.shape[0] == 10
    
    def test_ensemble_generation(self, generator, sequence):
        """Test ensemble generation."""
        structures, priors = generator.generate_ensemble(
            sequence, n_samples=5, temperature=0.1
        )
        
        assert len(structures) == 5
        assert all(s.shape == (10, 3) for s in structures)
    
    def test_temperature_effect(self, generator, sequence):
        """Test temperature affects diversity."""
        low_temp = generator.generate_ensemble(sequence, n_samples=10, temperature=0.01)
        high_temp = generator.generate_ensemble(sequence, n_samples=10, temperature=1.0)
        
        low_std = torch.std(torch.stack(low_temp[0]))
        high_std = torch.std(torch.stack(high_temp[0]))
        
        assert high_std > low_std
    
    def test_gradient_flow(self, generator, sequence):
        """Test gradients flow through generator."""
        sequence.requires_grad = False
        structure, _ = generator(sequence)
        loss = structure.sum()
        loss.backward()
        
        # Check gradients for parameters
        for param in generator.parameters():
            if param.requires_grad:
                assert param.grad is not None
