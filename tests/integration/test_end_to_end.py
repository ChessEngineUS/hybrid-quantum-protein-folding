"""End-to-end integration tests."""

import pytest
import torch
from hqpf.models.hybrid_model import HybridModel
from hqpf.data.benchmarks import load_benchmark_protein, AA_TO_IDX


class TestEndToEnd:
    """Test complete workflows."""
    
    def test_prediction_pipeline(self):
        """Test complete prediction pipeline."""
        # Load benchmark
        protein = load_benchmark_protein('peptide_10', dataset='test')
        sequence = protein.to_tensor(AA_TO_IDX, device='cpu')
        
        # Initialize model
        model = HybridModel(
            n_residues=len(sequence),
            n_qubits=len(sequence),
            use_quantum=False,
            device='cpu'
        )
        
        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(sequence, n_candidates=5)
        
        # Validate outputs
        assert outputs['best_structure'].shape == (len(sequence), 3)
        assert torch.all(torch.isfinite(outputs['best_structure']))
        assert torch.isfinite(outputs['best_energy'])
    
    def test_training_loop(self):
        """Test training loop integration."""
        sequence = torch.randint(0, 20, (10,), dtype=torch.long)
        
        model = HybridModel(
            n_residues=10,
            n_qubits=10,
            use_quantum=False,
            device='cpu'
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Training steps
        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(sequence, n_candidates=2)
            loss = outputs['best_energy']
            loss.backward()
            optimizer.step()
        
        # Check parameters updated
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert not torch.allclose(param, initial_params[name])
    
    def test_multi_protein_prediction(self):
        """Test predicting multiple proteins."""
        protein_ids = ['peptide_10', 'helix_12', 'sheet_10']
        
        for protein_id in protein_ids:
            protein = load_benchmark_protein(protein_id, dataset='test')
            sequence = protein.to_tensor(AA_TO_IDX, device='cpu')
            
            model = HybridModel(
                n_residues=len(sequence),
                n_qubits=len(sequence),
                use_quantum=False,
                device='cpu'
            )
            
            model.eval()
            with torch.no_grad():
                outputs = model(sequence, n_candidates=3)
            
            assert outputs['best_structure'].shape == (len(sequence), 3)
