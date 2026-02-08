"""Comprehensive tests for VQE Solver."""

import pytest
import torch
import numpy as np
from hqpf.models.vqe_solver import VQESolver


class TestVQESolver:
    """Test suite for VQE Solver quantum optimization."""
    
    @pytest.fixture
    def solver(self):
        """Create a VQE solver instance."""
        return VQESolver(
            n_qubits=4,
            ansatz_type='hardware_efficient',
            ansatz_depth=2,
            backend='statevector_simulator'
        )
    
    def test_initialization(self, solver):
        """Test VQE solver initialization."""
        assert solver.n_qubits == 4
        assert solver.ansatz_depth == 2
        assert solver.backend_name == 'statevector_simulator'
        assert solver.theta is not None
        assert len(solver.theta) > 0
    
    def test_parameter_count(self, solver):
        """Test correct parameter counting."""
        n_params = solver._count_parameters()
        expected_params = 4 * 2 * 2  # n_qubits * ansatz_depth * 2 (RY + RZ per layer)
        assert n_params == expected_params
        assert len(solver.theta) == expected_params
    
    def test_hamiltonian_construction(self, solver):
        """Test Hamiltonian matrix construction."""
        coords = torch.randn(4, 3)
        H = solver._construct_hamiltonian(coords)
        
        # Check Hamiltonian properties
        assert H.shape == (2**4, 2**4)
        assert torch.allclose(H, H.T)  # Hermitian
        assert torch.all(torch.isreal(H))  # Real for statevector
    
    def test_forward_pass(self, solver):
        """Test forward pass produces valid energy."""
        coords = torch.randn(4, 3)
        energy = solver(coords)
        
        assert isinstance(energy, torch.Tensor)
        assert energy.shape == ()
        assert torch.isfinite(energy)
        assert energy.requires_grad
    
    def test_gradient_flow(self, solver):
        """Test gradients flow through VQE."""
        coords = torch.randn(4, 3, requires_grad=True)
        energy = solver(coords)
        energy.backward()
        
        assert coords.grad is not None
        assert torch.any(coords.grad != 0)
        assert solver.theta.grad is not None
    
    def test_different_ansatz_types(self):
        """Test different ansatz constructions."""
        for ansatz in ['hardware_efficient', 'real_amplitudes']:
            solver = VQESolver(n_qubits=4, ansatz_type=ansatz, ansatz_depth=2)
            coords = torch.randn(4, 3)
            energy = solver(coords)
            assert torch.isfinite(energy)
    
    def test_batch_evaluation(self, solver):
        """Test batch processing."""
        batch_size = 8
        coords = torch.randn(batch_size, 4, 3)
        energies = torch.stack([solver(coords[i]) for i in range(batch_size)])
        
        assert energies.shape == (batch_size,)
        assert torch.all(torch.isfinite(energies))
    
    def test_energy_consistency(self, solver):
        """Test energy values are consistent."""
        coords = torch.randn(4, 3)
        energy1 = solver(coords)
        energy2 = solver(coords)
        
        assert torch.allclose(energy1, energy2, rtol=1e-5)
    
    def test_parameter_optimization(self, solver):
        """Test parameters can be optimized."""
        coords = torch.randn(4, 3)
        optimizer = torch.optim.Adam([solver.theta], lr=0.01)
        
        initial_energy = solver(coords).item()
        
        for _ in range(10):
            optimizer.zero_grad()
            energy = solver(coords)
            energy.backward()
            optimizer.step()
        
        final_energy = solver(coords).item()
        # Energy should change after optimization
        assert abs(final_energy - initial_energy) > 1e-6
    
    def test_cuda_compatibility(self, solver):
        """Test CUDA device compatibility."""
        if torch.cuda.is_available():
            solver = solver.to('cuda')
            coords = torch.randn(4, 3, device='cuda')
            energy = solver(coords)
            assert energy.device.type == 'cuda'
    
    @pytest.mark.parametrize('n_qubits', [2, 4, 6, 8])
    def test_scalability(self, n_qubits):
        """Test solver scales with qubit count."""
        solver = VQESolver(n_qubits=n_qubits, ansatz_depth=2)
        coords = torch.randn(n_qubits, 3)
        energy = solver(coords)
        assert torch.isfinite(energy)
