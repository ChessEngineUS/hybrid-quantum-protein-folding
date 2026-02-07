"""
Training infrastructure for hybrid quantum-classical protein folding.

Includes:
- Training loop with validation
- Checkpointing
- Logging and metrics tracking
- Benchmarking utilities
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
from datetime import datetime
from pathlib import Path
import logging

from ..models.hybrid_model import HybridModel
from ..data.dataset import ProteinDataset


class Trainer:
    """
    Training manager for hybrid model.
    
    Handles:
        - Training loop with alternating optimization
        - Validation and early stopping
        - Checkpointing and model saving
        - Metric tracking and logging
        - Surrogate refresh scheduling
        
    Parameters:
        model: HybridModel instance
        train_dataset: Training data
        val_dataset: Validation data
        config: Training configuration dictionary
    """
    
    def __init__(
        self,
        model: HybridModel,
        train_dataset: ProteinDataset,
        val_dataset: Optional[ProteinDataset] = None,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Default config
        default_config = {
            'batch_size': 1,  # Process one protein at a time
            'n_epochs': 100,
            'n_candidates': 10,
            'lr_generator': 1e-4,
            'lr_hamiltonian': 1e-5,
            'lr_surrogate': 1e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'validation_freq': 5,
            'checkpoint_freq': 10,
            'surrogate_refresh_freq': 100,
            'early_stopping_patience': 20,
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs'
        }
        self.config = {**default_config, **(config or {})}
        
        # Setup optimizers
        self.optimizer_generator = optim.Adam(
            self.model.generator.parameters(),
            lr=self.config['lr_generator'],
            weight_decay=self.config['weight_decay']
        )
        
        self.optimizer_hamiltonian = optim.Adam(
            self.model.hamiltonian.parameters(),
            lr=self.config['lr_hamiltonian'],
            weight_decay=self.config['weight_decay']
        )
        
        self.optimizer_surrogate = optim.Adam(
            self.model.surrogate.parameters(),
            lr=self.config['lr_surrogate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate schedulers
        self.scheduler_generator = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_generator, mode='min', factor=0.5, patience=10
        )
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmsd': [],
            'val_rmsd': []
        }
    
    def _setup_logging(self):
        """Setup logging infrastructure."""
        # Create directories
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['log_dir']).mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        log_file = os.path.join(
            self.config['log_dir'],
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Training configuration: {json.dumps(self.config, indent=2)}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            metrics_history: Dictionary of training metrics
        """
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['n_epochs']):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            if self.val_dataset is not None and epoch % self.config['validation_freq'] == 0:
                val_metrics = self._validate()
                self._log_metrics(train_metrics, val_metrics)
                
                # Early stopping check
                if self._check_early_stopping(val_metrics['loss']):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                self._log_metrics(train_metrics, None)
            
            # Checkpointing
            if epoch % self.config['checkpoint_freq'] == 0:
                self._save_checkpoint()
            
            # Learning rate scheduling
            if self.val_dataset is not None:
                self.scheduler_generator.step(val_metrics['loss'])
        
        self.logger.info("Training completed.")
        return self.metrics_history
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            metrics: Dictionary of epoch metrics
        """
        self.model.train()
        
        epoch_losses = []
        epoch_rmsds = []
        
        # DataLoader (batch_size=1 for proteins)
        for idx, data in enumerate(self.train_dataset):
            sequence = data['sequence'].to(self.model.device)
            targets = {
                k: v.to(self.model.device) if torch.is_tensor(v) else v
                for k, v in data.items() if k != 'sequence'
            }
            
            # Training step
            loss_dict = self.model.train_step(
                sequence=sequence,
                targets=targets,
                optimizer_generator=self.optimizer_generator,
                optimizer_hamiltonian=self.optimizer_hamiltonian,
                optimizer_surrogate=self.optimizer_surrogate,
                n_candidates=self.config['n_candidates']
            )
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.generator.parameters(),
                self.config['grad_clip']
            )
            
            epoch_losses.append(loss_dict['total'])
            if 'structure' in loss_dict:
                epoch_rmsds.append(loss_dict['structure'])
            
            # Logging
            if idx % 10 == 0:
                self.logger.info(
                    f"Epoch {self.epoch}, Batch {idx}: "
                    f"Loss={loss_dict['total']:.4f}, "
                    f"RMSD={loss_dict.get('structure', 0):.4f}"
                )
        
        # Epoch metrics
        metrics = {
            'loss': np.mean(epoch_losses),
            'rmsd': np.mean(epoch_rmsds) if epoch_rmsds else 0.0
        }
        
        return metrics
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """
        Validation loop.
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        
        val_losses = []
        val_rmsds = []
        
        for data in self.val_dataset:
            sequence = data['sequence'].to(self.model.device)
            targets = {
                k: v.to(self.model.device) if torch.is_tensor(v) else v
                for k, v in data.items() if k != 'sequence'
            }
            
            # Forward pass
            outputs = self.model(
                sequence,
                n_candidates=self.config['n_candidates'],
                use_surrogate=True
            )
            
            # Compute loss
            loss, loss_dict = self.model.compute_loss(outputs, targets)
            
            val_losses.append(loss.item())
            if 'structure' in loss_dict:
                val_rmsds.append(loss_dict['structure'])
        
        metrics = {
            'loss': np.mean(val_losses),
            'rmsd': np.mean(val_rmsds) if val_rmsds else 0.0
        }
        
        return metrics
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Optional[Dict]):
        """Log and store metrics."""
        # Update history
        self.metrics_history['train_loss'].append(train_metrics['loss'])
        self.metrics_history['train_rmsd'].append(train_metrics['rmsd'])
        
        log_msg = (
            f"Epoch {self.epoch}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Train RMSD={train_metrics['rmsd']:.4f}"
        )
        
        if val_metrics:
            self.metrics_history['val_loss'].append(val_metrics['loss'])
            self.metrics_history['val_rmsd'].append(val_metrics['rmsd'])
            log_msg += (
                f", Val Loss={val_metrics['loss']:.4f}, "
                f"Val RMSD={val_metrics['rmsd']:.4f}"
            )
        
        self.logger.info(log_msg)
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Check early stopping criterion.
        
        Returns:
            should_stop: Whether to stop training
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            # Save best model
            self._save_checkpoint(best=True)
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config['early_stopping_patience']:
                return True
            return False
    
    def _save_checkpoint(self, best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer_generator': self.optimizer_generator.state_dict(),
            'optimizer_hamiltonian': self.optimizer_hamiltonian.state_dict(),
            'optimizer_surrogate': self.optimizer_surrogate.state_dict(),
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        if best:
            path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
            self.logger.info(f"Saving best model to {path}")
        else:
            path = os.path.join(
                self.config['checkpoint_dir'],
                f'checkpoint_epoch_{self.epoch}.pt'
            )
            self.logger.info(f"Saving checkpoint to {path}")
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.model.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        self.optimizer_hamiltonian.load_state_dict(checkpoint['optimizer_hamiltonian'])
        self.optimizer_surrogate.load_state_dict(checkpoint['optimizer_surrogate'])
        self.epoch = checkpoint['epoch']
        self.metrics_history = checkpoint['metrics_history']
        
        self.logger.info(f"Resumed from epoch {self.epoch}")
    
    @torch.no_grad()
    def benchmark(
        self,
        test_dataset: ProteinDataset,
        metrics: List[str] = ['rmsd', 'tm_score', 'energy_ranking']
    ) -> Dict[str, float]:
        """
        Benchmark model on test set.
        
        Args:
            test_dataset: Test protein dataset
            metrics: List of metrics to compute
            
        Returns:
            results: Dictionary of benchmark results
        """
        self.model.eval()
        self.logger.info("Running benchmark...")
        
        results = {metric: [] for metric in metrics}
        
        for idx, data in enumerate(test_dataset):
            sequence = data['sequence'].to(self.model.device)
            targets = {
                k: v.to(self.model.device) if torch.is_tensor(v) else v
                for k, v in data.items() if k != 'sequence'
            }
            
            # Predict
            outputs = self.model(
                sequence,
                n_candidates=20,  # More candidates for benchmarking
                use_surrogate=True
            )
            
            predicted_structure = outputs['best_structure']
            
            # Compute metrics
            if 'rmsd' in metrics and 'native_structure' in targets:
                rmsd = self._compute_rmsd(predicted_structure, targets['native_structure'])
                results['rmsd'].append(rmsd)
            
            if 'tm_score' in metrics and 'native_structure' in targets:
                tm = self._compute_tm_score(predicted_structure, targets['native_structure'])
                results['tm_score'].append(tm)
            
            if 'energy_ranking' in metrics and 'stability_labels' in targets:
                ranking_acc = self._compute_ranking_accuracy(
                    outputs['energies'], targets['stability_labels']
                )
                results['energy_ranking'].append(ranking_acc)
            
            self.logger.info(f"Benchmark progress: {idx+1}/{len(test_dataset)}")
        
        # Aggregate results
        summary = {k: np.mean(v) for k, v in results.items() if len(v) > 0}
        
        self.logger.info(f"Benchmark results: {json.dumps(summary, indent=2)}")
        
        return summary
    
    def _compute_rmsd(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """Compute RMSD between predicted and target structures."""
        diff = pred - target
        rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))
        return rmsd.item()
    
    def _compute_tm_score(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Compute TM-score (simplified version).
        
        Full implementation would use proper alignment.
        """
        n_residues = pred.shape[0]
        d0 = 1.24 * (n_residues - 15) ** (1/3) - 1.8
        
        distances = torch.norm(pred - target, dim=1)
        tm_score = torch.mean(1.0 / (1.0 + (distances / d0) ** 2))
        
        return tm_score.item()
    
    def _compute_ranking_accuracy(
        self,
        energies: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Compute pairwise ranking accuracy.
        
        Args:
            energies: Predicted energies
            labels: Stability labels (higher = more stable)
            
        Returns:
            accuracy: Fraction of correctly ranked pairs
        """
        correct = 0
        total = 0
        
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] != labels[j]:
                    total += 1
                    # Check if ranking is correct
                    if (labels[i] > labels[j] and energies[i] < energies[j]) or \
                       (labels[i] < labels[j] and energies[i] > energies[j]):
                        correct += 1
        
        return correct / total if total > 0 else 0.0
