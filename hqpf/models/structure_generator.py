"""
Graph Neural Network (GNN) structure generator for protein conformations.

Generates candidate protein structures in coarse-grained space
using message-passing neural networks conditioned on sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple
import numpy as np


class ProteinGraphConv(MessagePassing):
    """
    Custom message-passing layer for protein graphs.
    
    Propagates information along backbone and contact edges.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')  # Aggregate messages by summation
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_msg = nn.Linear(2 * in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # x: (num_nodes, in_channels)
        # edge_index: (2, num_edges)
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        # x_i: (num_edges, in_channels) - target node features
        # x_j: (num_edges, in_channels) - source node features
        msg = torch.cat([x_i, x_j], dim=-1)
        return self.lin_msg(msg)
    
    def update(self, aggr_out, x):
        # aggr_out: (num_nodes, out_channels) - aggregated messages
        # x: (num_nodes, in_channels) - original node features
        return F.relu(self.lin(x) + aggr_out)


class StructureGenerator(nn.Module):
    """
    GNN-based structure generator: sequence → coarse-grained structure.
    
    Architecture:
    - Node embedding: amino acid type + positional encoding
    - Message passing: 3-5 layers propagating backbone + contact info
    - Decoder: MLP predicting lattice coordinates or dihedral angles
    
    Parameters:
        n_residues: Maximum number of residues
        embedding_dim: Node embedding dimension
        hidden_dim: Hidden layer dimension
        n_layers: Number of message-passing layers
        output_type: 'lattice' (3D coords) or 'dihedral' (phi, psi angles)
    """
    
    def __init__(
        self,
        n_residues: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 4,
        output_type: str = 'lattice',
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_residues = n_residues
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_type = output_type
        self.dropout = dropout
        
        # Node feature encoder
        self.aa_embedding = nn.Embedding(20, embedding_dim)  # 20 amino acids
        self.pos_embedding = nn.Embedding(n_residues, embedding_dim)  # Positional
        
        # Initial projection
        self.input_proj = nn.Linear(2 * embedding_dim, hidden_dim)
        
        # Message-passing layers
        self.conv_layers = nn.ModuleList([
            ProteinGraphConv(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Decoder: node features → coordinates/angles
        if output_type == 'lattice':
            output_dim = 3  # (x, y, z) lattice coordinates
        elif output_type == 'dihedral':
            output_dim = 2  # (phi, psi) angles
        else:
            raise ValueError(f"Unknown output type: {output_type}")
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Prior network for regularization
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        sequence: torch.Tensor,
        return_prior: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate structure from sequence.
        
        Args:
            sequence: Integer tensor of shape (n_residues,) with AA indices
            return_prior: Whether to return prior probability
            
        Returns:
            structure: (n_residues, output_dim) coordinates or angles
            prior: Optional prior log-probability
        """
        n_residues = sequence.shape[0]
        
        # Embed sequence
        aa_emb = self.aa_embedding(sequence)  # (n_residues, embedding_dim)
        pos_emb = self.pos_embedding(torch.arange(n_residues, device=sequence.device))
        
        # Combine embeddings
        node_features = torch.cat([aa_emb, pos_emb], dim=-1)  # (n_residues, 2*embedding_dim)
        node_features = self.input_proj(node_features)  # (n_residues, hidden_dim)
        
        # Construct graph: backbone edges (sequential) + long-range contacts
        edge_index = self._construct_edges(n_residues, sequence.device)
        
        # Message passing
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            node_features_new = conv(node_features, edge_index)
            node_features_new = norm(node_features_new)
            # Residual connection
            if i > 0:
                node_features = node_features + F.dropout(node_features_new, p=self.dropout, training=self.training)
            else:
                node_features = node_features_new
        
        # Decode to structure
        structure = self.decoder(node_features)  # (n_residues, output_dim)
        
        # Apply constraints
        if self.output_type == 'lattice':
            # Discretize to lattice grid (3.8 Å spacing)
            structure = torch.round(structure / 3.8) * 3.8
        elif self.output_type == 'dihedral':
            # Constrain to [-π, π]
            structure = torch.tanh(structure) * np.pi
        
        # Compute prior if requested
        prior = None
        if return_prior:
            # Global pooling + prior network
            global_features = torch.mean(node_features, dim=0)  # (hidden_dim,)
            prior = self.prior_net(global_features).squeeze()  # scalar
        
        return structure, prior
    
    def _construct_edges(self, n_residues: int, device: torch.device) -> torch.Tensor:
        """
        Construct protein graph edges.
        
        Includes:
        - Backbone edges: (i, i+1) for all i
        - Long-range edges: (i, j) for |i-j| > 3 (potential contacts)
        
        Returns:
            edge_index: (2, num_edges) tensor
        """
        edges = []
        
        # Backbone edges (bidirectional)
        for i in range(n_residues - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        
        # Long-range contact edges (sample subset)
        max_contacts = min(20, n_residues // 2)  # Limit for efficiency
        for i in range(n_residues):
            # Sample long-range neighbors
            candidates = [j for j in range(n_residues) if abs(j - i) > 3]
            if len(candidates) > 0:
                sampled = np.random.choice(candidates, size=min(3, len(candidates)), replace=False)
                for j in sampled:
                    edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        return edge_index
    
    def generate_ensemble(
        self,
        sequence: torch.Tensor,
        n_samples: int = 10,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate ensemble of structures via stochastic sampling.
        
        Args:
            sequence: Amino acid sequence
            n_samples: Number of structures to generate
            temperature: Sampling temperature (higher = more diverse)
            
        Returns:
            ensemble: (n_samples, n_residues, output_dim)
        """
        ensemble = []
        for _ in range(n_samples):
            # Add noise to embeddings for stochasticity
            structure, _ = self.forward(sequence)
            if temperature > 0:
                noise = torch.randn_like(structure) * temperature
                structure = structure + noise
            ensemble.append(structure)
        
        return torch.stack(ensemble)
