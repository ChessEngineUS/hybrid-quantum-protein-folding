"""
Hybrid Quantum-Classical Model for Protein Structure Prediction.

Integrates:
- GNN structure generator
- Quantum Hamiltonian (VQE)
- CNN surrogate model
- End-to-end training with hybrid loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .structure_generator import StructureGenerator
from .vqe_solver import VQESolver
from .surrogate import SurrogateModel
from .quantum_hamiltonian import QuantumHamiltonian


class HybridModel(nn.Module):
    """
    End-to-end hybrid quantum-classical protein structure prediction.
    
    Architecture:
        1. GNN generates candidate structures from sequence
        2. Quantum VQE (or surrogate) evaluates energy landscape
        3. Hybrid loss trains both generator and Hamiltonian parameters
        
    Training strategy:
        - Sample 10% of candidates for quantum evaluation
        - Use surrogate for remaining 90%
        - Periodically refresh surrogate with new quantum data
        
    Parameters:
        n_residues: Maximum protein length
        use_quantum: Whether to use real quantum hardware (vs simulator)
        surrogate_refresh_freq: Iterations between surrogate updates
    """
    
    def __init__(
        self,
        n_residues: int,
        n_qubits: int = 20,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        n_gnn_layers: int = 4,
        n_vqe_layers: int = 3,
        use_quantum: bool = False,
        backend: str = 'qiskit',
        surrogate_refresh_freq: int = 100,
        quantum_fraction: float = 0.1,
        output_type: str = 'lattice',
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.n_residues = n_residues
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        self.surrogate_refresh_freq = surrogate_refresh_freq
        self.quantum_fraction = quantum_fraction
        self.output_type = output_type
        self.device = device
        
        # Structure generator (GNN)
        self.generator = StructureGenerator(
            n_residues=n_residues,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_gnn_layers,
            output_type=output_type
        ).to(device)
        
        # Quantum Hamiltonian
        self.hamiltonian = QuantumHamiltonian(
            n_residues=n_residues,
            n_qubits=n_qubits
        ).to(device)
        
        # VQE solver
        self.vqe_solver = VQESolver(
            n_qubits=n_qubits,
            n_layers=n_vqe_layers,
            backend=backend,
            use_real_backend=use_quantum
        )
        
        # Surrogate model
        input_channels = 10  # AA features + structure + contacts + curvature
        self.surrogate = SurrogateModel(
            n_residues=n_residues,
            input_channels=input_channels,
            hidden_channels=64,
            n_conv_layers=3
        ).to(device)
        
        # Training state
        self.iteration = 0
        self.quantum_cache = []  # Cache for surrogate training
        self.cache_size = 200
        
    def forward(
        self,
        sequence: torch.Tensor,
        n_candidates: int = 10,
        use_surrogate: bool = True,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: sequence → structures → energies → ranking.
        
        Args:
            sequence: (n_residues,) amino acid indices
            n_candidates: Number of candidate structures to generate
            use_surrogate: Use surrogate for energy (vs quantum)
            temperature: Sampling temperature for ensemble generation
            
        Returns:
            Dictionary containing:
                - structures: (n_candidates, n_residues, output_dim)
                - energies: (n_candidates,)
                - prior: (n_candidates,) prior log-probs from GNN
                - best_structure: (n_residues, output_dim)
                - quantum_fraction: Fraction evaluated on quantum hardware
        """
        batch_size = n_candidates
        n_res = sequence.shape[0]
        
        # 1. Generate candidate structures
        if n_candidates == 1:
            structure, prior = self.generator(sequence, return_prior=True)
            structures = structure.unsqueeze(0)
            priors = prior.unsqueeze(0) if prior is not None else None
        else:
            structures = self.generator.generate_ensemble(
                sequence, n_samples=n_candidates, temperature=temperature
            )
            priors = []
            for i in range(n_candidates):
                _, p = self.generator(sequence, return_prior=True)
                priors.append(p if p is not None else torch.tensor(0.0))
            priors = torch.stack(priors) if len(priors) > 0 else None
        
        # 2. Evaluate energies (quantum + classical)
        energies_total = []
        quantum_mask = []
        
        # Decide which structures to evaluate on quantum hardware
        if use_surrogate and self.training:
            # During training: sample quantum_fraction for quantum, rest for surrogate
            n_quantum = max(1, int(n_candidates * self.quantum_fraction))
            quantum_indices = np.random.choice(n_candidates, size=n_quantum, replace=False)
            quantum_mask = torch.zeros(n_candidates, dtype=torch.bool)
            quantum_mask[quantum_indices] = True
        else:
            # Inference: use surrogate if available and requested
            if use_surrogate:
                quantum_mask = torch.zeros(n_candidates, dtype=torch.bool)
            else:
                quantum_mask = torch.ones(n_candidates, dtype=torch.bool)
        
        # Evaluate energies
        for i in range(n_candidates):
            structure = structures[i]
            
            if quantum_mask[i]:
                # Quantum evaluation
                energy = self._evaluate_quantum(sequence, structure)
                
                # Cache for surrogate training
                if self.training and len(self.quantum_cache) < self.cache_size:
                    self.quantum_cache.append({
                        'sequence': sequence.detach().cpu(),
                        'structure': structure.detach().cpu(),
                        'energy': energy.detach().cpu()
                    })
            else:
                # Surrogate evaluation
                energy = self._evaluate_surrogate(sequence, structure)
            
            energies_total.append(energy)
        
        energies_total = torch.stack(energies_total)
        
        # 3. Compute classical energy components
        energies_classical = self._evaluate_classical_batch(sequence, structures)
        
        # 4. Hybrid energy fusion
        lambda_weight = 0.5  # Tunable weight
        energies_hybrid = energies_classical + lambda_weight * energies_total
        
        # 5. Add prior term
        if priors is not None:
            energies_hybrid = energies_hybrid + 0.1 * priors  # Regularization weight
        
        # 6. Select best structure
        best_idx = torch.argmin(energies_hybrid)
        best_structure = structures[best_idx]
        
        return {
            'structures': structures,
            'energies': energies_hybrid,
            'energies_quantum': energies_total,
            'energies_classical': energies_classical,
            'prior': priors,
            'best_structure': best_structure,
            'best_energy': energies_hybrid[best_idx],
            'quantum_fraction': quantum_mask.float().mean().item()
        }
    
    def _evaluate_quantum(
        self,
        sequence: torch.Tensor,
        structure: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate structure using quantum VQE."""
        # Build Hamiltonian for this conformation
        hamiltonian_matrix = self.hamiltonian(sequence, structure)
        
        # Convert to Pauli operators
        pauli_hamiltonian = self.hamiltonian.to_pauli_operators(hamiltonian_matrix)
        
        # Run VQE
        result = self.vqe_solver.run(pauli_hamiltonian)
        
        return torch.tensor(result['energy'], dtype=torch.float32, device=self.device)
    
    def _evaluate_surrogate(
        self,
        sequence: torch.Tensor,
        structure: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate structure using surrogate model."""
        # Encode conformation
        contact_map = self._compute_contact_map(structure)
        conformation = self.surrogate.encode_conformation(
            sequence, structure, contact_map
        )
        
        # Predict energy
        energy, _ = self.surrogate(conformation.unsqueeze(0))
        
        return energy.squeeze(0)
    
    def _evaluate_classical_batch(
        self,
        sequence: torch.Tensor,
        structures: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate classical energy terms for batch of structures.
        
        Includes: hydrophobic effect, H-bonds, entropy
        """
        n_candidates = structures.shape[0]
        energies = []
        
        for i in range(n_candidates):
            structure = structures[i]
            
            # Hydrophobic effect
            e_hydro = self._hydrophobic_energy(sequence, structure)
            
            # H-bond energy (simplified)
            e_hbond = self._hbond_energy(structure)
            
            # Entropic penalty (compactness)
            e_entropy = self._entropy_penalty(structure)
            
            energy_classical = e_hydro + e_hbond + e_entropy
            energies.append(energy_classical)
        
        return torch.stack(energies)
    
    def _hydrophobic_energy(
        self,
        sequence: torch.Tensor,
        structure: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hydrophobic effect energy.
        
        Penalizes hydrophilic residues in buried positions.
        """
        # Hydrophobicity scale (simplified)
        hydro_scale = torch.tensor([
            0.62, -0.90, -0.74, -0.81, 1.07,  # A, R, N, D, C
            -0.64, 0.48, -0.31, 1.38, -0.78,  # Q, E, G, H, I
            0.97, -1.50, 0.85, 0.67, -0.85,   # L, K, M, F, P
            -0.42, -0.98, 0.07, 1.53, 0.22    # S, T, W, Y, V
        ], device=self.device)
        
        hydro_values = hydro_scale[sequence]  # (n_residues,)
        
        # Compute burial (contact count)
        contact_map = self._compute_contact_map(structure)
        burial = contact_map.sum(dim=1)  # (n_residues,)
        burial_normalized = burial / burial.max()
        
        # Energy: hydrophobic residues prefer buried, hydrophilic prefer surface
        energy = -torch.sum(hydro_values * burial_normalized)
        
        return energy
    
    def _hbond_energy(self, structure: torch.Tensor) -> torch.Tensor:
        """
        Compute H-bond energy (simplified geometric criterion).
        """
        n_residues = structure.shape[0]
        energy = torch.tensor(0.0, device=self.device)
        
        # H-bond if distance < 5 Å and |i-j| > 3
        for i in range(n_residues):
            for j in range(i + 4, n_residues):
                dist = torch.norm(structure[i] - structure[j])
                if dist < 5.0:
                    energy -= 1.0  # H-bond strength
        
        return energy
    
    def _entropy_penalty(self, structure: torch.Tensor) -> torch.Tensor:
        """
        Entropic penalty: favors compact structures.
        """
        # Radius of gyration
        center = structure.mean(dim=0)
        rg = torch.sqrt(torch.mean(torch.sum((structure - center) ** 2, dim=1)))
        
        # Penalty proportional to Rg
        return 0.1 * rg
    
    def _compute_contact_map(
        self,
        structure: torch.Tensor,
        cutoff: float = 8.0
    ) -> torch.Tensor:
        """
        Compute binary contact map based on distance cutoff.
        
        Args:
            structure: (n_residues, 3 or 2)
            cutoff: Distance threshold in Å
            
        Returns:
            contact_map: (n_residues, n_residues) binary matrix
        """
        n_residues = structure.shape[0]
        
        # Pairwise distances
        dist_matrix = torch.cdist(structure, structure, p=2)
        
        # Binary contacts
        contact_map = (dist_matrix < cutoff).float()
        
        # Remove diagonal
        contact_map.fill_diagonal_(0)
        
        return contact_map
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Dict[str, float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid loss function.
        
        Loss components:
            1. Structural loss: RMSD to native or contact accuracy
            2. Energy ranking loss: Rank native above decoys
            3. Prior regularization: Encourage realistic structures
            
        Args:
            outputs: Model outputs from forward()
            targets: Ground truth (native structure, stability labels, etc.)
            weights: Loss component weights
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Individual loss components
        """
        if weights is None:
            weights = {'structure': 1.0, 'energy': 1.0, 'prior': 0.1}
        
        loss_dict = {}
        
        # 1. Structural loss
        if 'native_structure' in targets:
            native = targets['native_structure']
            predicted = outputs['best_structure']
            
            # RMSD loss (after alignment)
            rmsd = self._rmsd_loss(predicted, native)
            loss_dict['structure'] = rmsd.item()
            loss_structure = weights['structure'] * rmsd
        else:
            loss_structure = torch.tensor(0.0, device=self.device)
        
        # 2. Energy ranking loss
        if 'stability_labels' in targets:
            # stability_labels: (n_candidates,) with 1=stable, 0=unstable
            labels = targets['stability_labels']
            energies = outputs['energies']
            
            # Pairwise ranking loss: stable should have lower energy
            loss_ranking = torch.tensor(0.0, device=self.device)
            n_pairs = 0
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if labels[i] > labels[j]:  # i is more stable
                        loss_ranking += F.relu(energies[i] - energies[j] + 1.0)  # Margin=1.0
                        n_pairs += 1
                    elif labels[j] > labels[i]:
                        loss_ranking += F.relu(energies[j] - energies[i] + 1.0)
                        n_pairs += 1
            
            if n_pairs > 0:
                loss_ranking /= n_pairs
            
            loss_dict['ranking'] = loss_ranking.item()
            loss_energy = weights['energy'] * loss_ranking
        elif 'delta_G' in targets:
            # Regression on experimental ΔG
            delta_G_pred = outputs['best_energy']
            delta_G_true = targets['delta_G']
            
            loss_energy = weights['energy'] * F.mse_loss(delta_G_pred, delta_G_true)
            loss_dict['energy'] = loss_energy.item()
        else:
            loss_energy = torch.tensor(0.0, device=self.device)
        
        # 3. Prior regularization
        if outputs['prior'] is not None:
            # Encourage high prior probability for generated structures
            loss_prior = -torch.mean(outputs['prior'])  # Maximize log-prob
            loss_dict['prior'] = loss_prior.item()
            loss_prior = weights['prior'] * loss_prior
        else:
            loss_prior = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = loss_structure + loss_energy + loss_prior
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _rmsd_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RMSD after optimal alignment (Kabsch algorithm).
        
        For simplicity, using unaligned RMSD here.
        Production version should implement Kabsch.
        """
        # Unaligned RMSD
        diff = pred - target
        rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))
        
        return rmsd
    
    def update_surrogate(self, optimizer: torch.optim.Optimizer) -> float:
        """
        Update surrogate model with cached quantum evaluations.
        
        Args:
            optimizer: Optimizer for surrogate parameters
            
        Returns:
            loss: Training loss
        """
        if len(self.quantum_cache) < 10:
            return 0.0  # Not enough data
        
        # Sample batch from cache
        batch_size = min(32, len(self.quantum_cache))
        indices = np.random.choice(len(self.quantum_cache), size=batch_size, replace=False)
        
        batch_sequences = []
        batch_structures = []
        batch_energies = []
        
        for idx in indices:
            data = self.quantum_cache[idx]
            batch_sequences.append(data['sequence'])
            batch_structures.append(data['structure'])
            batch_energies.append(data['energy'])
        
        # Prepare inputs
        conformations = []
        for seq, struct in zip(batch_sequences, batch_structures):
            seq = seq.to(self.device)
            struct = struct.to(self.device)
            contact_map = self._compute_contact_map(struct)
            conf = self.surrogate.encode_conformation(seq, struct, contact_map)
            conformations.append(conf)
        
        conformations = torch.stack(conformations)  # (batch, n_residues, features)
        energies = torch.stack([e.to(self.device) for e in batch_energies])
        
        # Update surrogate
        loss = self.surrogate.update_with_quantum_data(
            energies, conformations, optimizer
        )
        
        return loss
    
    def train_step(
        self,
        sequence: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        optimizer_generator: torch.optim.Optimizer,
        optimizer_hamiltonian: torch.optim.Optimizer,
        optimizer_surrogate: torch.optim.Optimizer,
        n_candidates: int = 10
    ) -> Dict[str, float]:
        """
        Single training step with alternating optimization.
        
        Steps:
            1. Generate structures (GNN)
            2. Evaluate energies (quantum + surrogate)
            3. Compute loss and update generator
            4. Periodically update Hamiltonian parameters
            5. Periodically refresh surrogate
        """
        self.train()
        self.iteration += 1
        
        # Forward pass
        outputs = self.forward(
            sequence,
            n_candidates=n_candidates,
            use_surrogate=True,
            temperature=1.0
        )
        
        # Compute loss
        total_loss, loss_dict = self.compute_loss(outputs, targets)
        
        # Backward pass - update generator
        optimizer_generator.zero_grad()
        total_loss.backward()
        optimizer_generator.step()
        
        # Update Hamiltonian parameters (less frequently)
        if self.iteration % 10 == 0:
            # Collect high-confidence samples
            # Re-optimize Hamiltonian to match experimental stabilities
            # (Placeholder: would require additional supervised regression)
            pass
        
        # Update surrogate (periodically)
        if self.iteration % self.surrogate_refresh_freq == 0:
            loss_surr = self.update_surrogate(optimizer_surrogate)
            loss_dict['surrogate'] = loss_surr
        
        return loss_dict
