"""
Parametric Quantum Hamiltonian for protein conformational energy.

Implements H_eff(θ) = H_local(θ) + H_contact(θ) + H_env(θ)
with learnable parameters θ conditioned on sequence and environment.
"""

import numpy as np
import torch
import torch.nn as nn
from qiskit.quantum_info import SparsePauliOp
from typing import Dict, List, Tuple, Optional


class QuantumHamiltonian(nn.Module):
    """
    Learnable quantum Hamiltonian for protein energy landscapes.
    
    Parameters:
        n_qubits: Number of qubits encoding the protein conformation
        n_residues: Number of residues in the protein fragment
        sequence_embedding_dim: Dimension of sequence embeddings
        learn_local: Whether to learn local (dihedral) terms
        learn_contact: Whether to learn non-local contact terms
        learn_environment: Whether to learn environment-dependent terms
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_residues: int,
        sequence_embedding_dim: int = 64,
        learn_local: bool = True,
        learn_contact: bool = True,
        learn_environment: bool = True,
        max_contact_distance: int = 4,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_residues = n_residues
        self.sequence_embedding_dim = sequence_embedding_dim
        self.learn_local = learn_local
        self.learn_contact = learn_contact
        self.learn_environment = learn_environment
        self.max_contact_distance = max_contact_distance
        
        # Sequence encoder: maps amino acid sequence to embeddings
        self.sequence_encoder = nn.Sequential(
            nn.Embedding(20, sequence_embedding_dim),  # 20 amino acids
            nn.Linear(sequence_embedding_dim, sequence_embedding_dim),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim, sequence_embedding_dim)
        )
        
        # Local term parameters: J^(φ)_i for dihedral preferences
        if learn_local:
            self.local_weight_net = nn.Sequential(
                nn.Linear(sequence_embedding_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # Output: [J_ZZ, J_XY] coefficients
            )
        
        # Contact term parameters: K_ij for residue pair interactions
        if learn_contact:
            self.contact_weight_net = nn.Sequential(
                nn.Linear(sequence_embedding_dim * 2 + 3, 128),  # +3 for distance, hydrophobicity, charge
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # Output: K_ij
            )
            
            # Hydrophobicity scale (Kyte-Doolittle)
            self.register_buffer('hydrophobicity', torch.tensor([
                1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
                3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2
            ]))  # A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
            
            # Charge (+1, 0, -1)
            self.register_buffer('charge', torch.tensor([
                0, 1, 0, -1, 0, 0, -1, 0, 0.5, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0
            ]))
        
        # Environment term parameters: α_i for pH, β_ij for force
        if learn_environment:
            self.env_ph_net = nn.Linear(sequence_embedding_dim, 1)
            self.env_force_net = nn.Sequential(
                nn.Linear(sequence_embedding_dim * 2 + 1, 64),  # +1 for force magnitude
                nn.ReLU(),
                nn.Linear(64, 1)
            )
    
    def forward(
        self,
        sequence: torch.Tensor,
        conformation: torch.Tensor,
        environment: Optional[Dict[str, torch.Tensor]] = None
    ) -> SparsePauliOp:
        """
        Construct the quantum Hamiltonian for a given sequence and conformation.
        
        Args:
            sequence: Integer tensor of shape (n_residues,) with amino acid indices
            conformation: Lattice positions or dihedral angles, shape (n_residues, 3 or 2)
            environment: Optional dict with 'ph', 'force', 'ionic_strength'
            
        Returns:
            SparsePauliOp representing the Hamiltonian
        """
        # Encode sequence
        seq_embeddings = self.sequence_encoder(sequence)  # (n_residues, embedding_dim)
        
        # Initialize Hamiltonian as list of Pauli operators
        pauli_strings = []
        coefficients = []
        
        # Add local terms
        if self.learn_local:
            local_ops, local_coeffs = self._construct_local_terms(
                seq_embeddings, conformation
            )
            pauli_strings.extend(local_ops)
            coefficients.extend(local_coeffs)
        
        # Add contact terms
        if self.learn_contact:
            contact_ops, contact_coeffs = self._construct_contact_terms(
                seq_embeddings, sequence, conformation
            )
            pauli_strings.extend(contact_ops)
            coefficients.extend(contact_coeffs)
        
        # Add environment terms
        if self.learn_environment and environment is not None:
            env_ops, env_coeffs = self._construct_environment_terms(
                seq_embeddings, conformation, environment
            )
            pauli_strings.extend(env_ops)
            coefficients.extend(env_coeffs)
        
        # Convert to numpy for Qiskit
        coefficients_np = np.array([c.detach().cpu().numpy() for c in coefficients])
        
        # Construct SparsePauliOp
        hamiltonian = SparsePauliOp(pauli_strings, coefficients_np)
        
        return hamiltonian
    
    def _construct_local_terms(
        self,
        seq_embeddings: torch.Tensor,
        conformation: torch.Tensor
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Construct local backbone dihedral terms:
        H_local = Σ_i [J^(φ)_i σ^Z_i σ^Z_{i+1} + Γ^(ψ)_i (σ^X_i + σ^Y_i)]
        """
        pauli_strings = []
        coefficients = []
        
        for i in range(self.n_residues - 1):
            # Concatenate neighboring residue embeddings
            pair_embedding = torch.cat([seq_embeddings[i], seq_embeddings[i+1]], dim=0)
            
            # Predict coupling strengths
            weights = self.local_weight_net(pair_embedding)  # (2,)
            j_zz = weights[0]
            j_xy = weights[1]
            
            # Map residues to qubits (assume 2 qubits per residue for encoding dihedrals)
            qubit_i = 2 * i
            qubit_i1 = 2 * (i + 1)
            
            # ZZ term for dihedral coupling
            pauli_str_zz = 'I' * qubit_i + 'Z' + 'I' * (qubit_i1 - qubit_i - 1) + 'Z' + 'I' * (self.n_qubits - qubit_i1 - 1)
            pauli_strings.append(pauli_str_zz)
            coefficients.append(j_zz)
            
            # X + Y term for transverse field
            pauli_str_x = 'I' * qubit_i + 'X' + 'I' * (self.n_qubits - qubit_i - 1)
            pauli_str_y = 'I' * qubit_i + 'Y' + 'I' * (self.n_qubits - qubit_i - 1)
            pauli_strings.append(pauli_str_x)
            coefficients.append(j_xy)
            pauli_strings.append(pauli_str_y)
            coefficients.append(j_xy)
        
        return pauli_strings, coefficients
    
    def _construct_contact_terms(
        self,
        seq_embeddings: torch.Tensor,
        sequence: torch.Tensor,
        conformation: torch.Tensor
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Construct non-local contact terms:
        H_contact = Σ_{i<j, |i-j|>2} K_ij(θ) Z_i Z_j
        """
        pauli_strings = []
        coefficients = []
        
        for i in range(self.n_residues):
            for j in range(i + 3, min(i + self.max_contact_distance + 1, self.n_residues)):
                # Compute distance between residues
                if conformation.shape[1] == 3:  # Lattice coordinates
                    distance = torch.norm(conformation[i] - conformation[j])
                else:  # Dihedral angles - use sequence separation as proxy
                    distance = torch.tensor(float(j - i))
                
                # Get residue properties
                aa_i = sequence[i]
                aa_j = sequence[j]
                hydro_i = self.hydrophobicity[aa_i]
                hydro_j = self.hydrophobicity[aa_j]
                charge_i = self.charge[aa_i]
                charge_j = self.charge[aa_j]
                
                # Feature vector for contact
                contact_features = torch.cat([
                    seq_embeddings[i],
                    seq_embeddings[j],
                    distance.unsqueeze(0),
                    (hydro_i * hydro_j).unsqueeze(0),
                    (charge_i * charge_j).unsqueeze(0)
                ])
                
                # Predict contact strength
                k_ij = self.contact_weight_net(contact_features).squeeze()
                
                # Map to qubits
                qubit_i = 2 * i
                qubit_j = 2 * j
                
                # ZZ interaction
                pauli_str = 'I' * qubit_i + 'Z' + 'I' * (qubit_j - qubit_i - 1) + 'Z' + 'I' * (self.n_qubits - qubit_j - 1)
                pauli_strings.append(pauli_str)
                coefficients.append(k_ij)
        
        return pauli_strings, coefficients
    
    def _construct_environment_terms(
        self,
        seq_embeddings: torch.Tensor,
        conformation: torch.Tensor,
        environment: Dict[str, torch.Tensor]
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Construct environment-dependent terms:
        H_env = Σ_i α_i(pH) Z_i + Σ_{i,j} β_ij(Force) Z_i Z_j
        """
        pauli_strings = []
        coefficients = []
        
        # pH-dependent terms (affects charged residues)
        if 'ph' in environment:
            ph_value = environment['ph']
            for i in range(self.n_residues):
                alpha_i = self.env_ph_net(seq_embeddings[i]).squeeze()
                alpha_i = alpha_i * ph_value  # Scale by pH
                
                qubit_i = 2 * i
                pauli_str = 'I' * qubit_i + 'Z' + 'I' * (self.n_qubits - qubit_i - 1)
                pauli_strings.append(pauli_str)
                coefficients.append(alpha_i)
        
        # Force-dependent terms (affects end-to-end distance)
        if 'force' in environment:
            force_magnitude = environment['force']
            for i in range(self.n_residues - 1):
                for j in range(i + 1, self.n_residues):
                    force_features = torch.cat([
                        seq_embeddings[i],
                        seq_embeddings[j],
                        force_magnitude.unsqueeze(0)
                    ])
                    beta_ij = self.env_force_net(force_features).squeeze()
                    
                    qubit_i = 2 * i
                    qubit_j = 2 * j
                    pauli_str = 'I' * qubit_i + 'Z' + 'I' * (qubit_j - qubit_i - 1) + 'Z' + 'I' * (self.n_qubits - qubit_j - 1)
                    pauli_strings.append(pauli_str)
                    coefficients.append(beta_ij)
        
        return pauli_strings, coefficients
    
    def get_parameter_summary(self) -> Dict[str, int]:
        """Return a summary of learnable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'local_parameters': sum(p.numel() for p in self.local_weight_net.parameters()) if self.learn_local else 0,
            'contact_parameters': sum(p.numel() for p in self.contact_weight_net.parameters()) if self.learn_contact else 0,
            'environment_parameters': sum(p.numel() for p in [*self.env_ph_net.parameters(), *self.env_force_net.parameters()]) if self.learn_environment else 0,
        }
