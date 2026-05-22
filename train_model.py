#!/usr/bin/env python3
"""
Model Training Module for AI Protein Design Pipeline
Handles training of protein sequence generation models
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from models.utils.model_architectures import (
    ProteinGenerator, 
    ProteinDiscriminator, 
    ProteinProGAN, 
    ProteinVAE
)
from models.utils.model_loader import ModelLoader
from models.utils.model_saver import ModelSaver
from datasets.utils.protein_preprocessor import ProteinPreprocessor

class ProteinDataset(Dataset):
    """Dataset class for protein sequences"""
    
    def __init__(
        self, 
        sequences: List[str], 
        vocab_size: int = 21, 
        max_length: int = 512
    ):
        self.sequences = sequences
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Create amino acid to index mapping
        self.aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15,
            'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }
        self.idx_to_aa = {v: k for k, v in self.aa_to_idx.items()}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Convert to tensor
        tensor = torch.zeros(self.max_length, dtype=torch.long)
        
        # Convert amino acids to indices
        for i, aa in enumerate(sequence):
            if i >= self.max_length:
                break
            if aa in self.aa_to_idx:
                tensor[i] = self.aa_to_idx[aa]
            else:
                tensor[i] = 0  # Default to 'A'
        
        return tensor

class ProteinProGANTrainer:
    """Trainer class for ProGAN protein sequence generation"""
    
    def __init__(
        self,
        config: Dict,
        device: str = "cpu"
    ):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.generator = ProteinGenerator(
            latent_dim=config.get('latent_dim', 128),
            vocab_size=config.get('vocab_size', 21),
            max_length=config.get('max_length', 512)
        ).to(device)
        
        self.discriminator = ProteinDiscriminator(
            vocab_size=config.get('vocab_size', 21),
            max_length=config.get('max_length', 512)
        ).to(device)
        
        # Initialize optimizers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=config.get('gen_lr', 0.0002),
            betas=(0.5, 0.999)
        )
        
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=config.get('disc_lr', 0.0002),
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # Training history
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'epoch': []
        }
        
        # Model saver
        self.model_saver = ModelSaver()
    
    def train_step(self, real_sequences: torch.Tensor, batch_size: int) -> Dict[str, float]:
        """Single training step"""
        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)
        
        # Generate noise
        noise = torch.randn(batch_size, self.config.get('latent_dim', 128), device=self.device)
        
        # Train Generator
        self.gen_optimizer.zero_grad()
        
        # Generate sequences
        generated_sequences = self.generator(noise, return_sequences=True)
        
        # Generator loss
        gen_validity = self.discriminator(generated_sequences)
        gen_loss = self.adversarial_loss(gen_validity, valid)
        
        gen_loss.backward()
        self.gen_optimizer.step()
        
        # Train Discriminator
        self.disc_optimizer.zero_grad()
        
        # Real loss
        real_validity = self.discriminator(real_sequences)
        real_loss = self.adversarial_loss(real_validity, valid)
        
        # Fake loss
        fake_validity = self.discriminator(generated_sequences.detach())
        fake_loss = self.adversarial_loss(fake_validity, fake)
        
        # Total discriminator loss
        disc_loss = (real_loss + fake_loss) / 2
        
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return {
            'generator_loss': gen_loss.item(),
            'discriminator_loss': disc_loss.item()
        }
    
    def train(
        self,
        train_sequences: List[str],
        epochs: int,
        batch_size: int = 32,
        save_every: int = 10,
        model_name: str = "progan_training"
    ) -> Dict[str, float]:
        """Train the ProGAN model"""
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # Create dataset and dataloader
        dataset = ProteinDataset(train_sequences)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            
            # Training loop
            for batch_idx, real_sequences in enumerate(dataloader):
                real_sequences = real_sequences.to(self.device)
                
                # Train step
                losses = self.train_step(real_sequences, real_sequences.size(0))
                
                epoch_gen_loss += losses['generator_loss']
                epoch_disc_loss += losses['discriminator_loss']
            
            # Calculate average losses
            num_batches = len(dataloader)
            avg_gen_loss = epoch_gen_loss / num_batches
            avg_disc_loss = epoch_disc_loss / num_batches
            
            # Update history
            self.training_history['generator_loss'].append(avg_gen_loss)
            self.training_history['discriminator_loss'].append(avg_disc_loss)
            self.training_history['epoch'].append(epoch)
            
            # Log progress
            if epoch % 5 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{epochs} - Gen Loss: {avg_gen_loss:.4f}, "
                    f"Disc Loss: {avg_disc_loss:.4f}"
                )
            
            # Save checkpoint
            if epoch % save_every == 0 or epoch == epochs - 1:
                is_best = avg_gen_loss < best_loss
                if is_best:
                    best_loss = avg_gen_loss
                
                # Create ProGAN wrapper
                progan = ProteinProGAN(
                    vocab_size=self.config.get('vocab_size', 21),
                    latent_dim=self.config.get('latent_dim', 128)
                )
                progan.generator = self.generator
                progan.discriminator = self.discriminator
                
                # Save checkpoint
                checkpoint_path = self.model_saver.save_checkpoint(
                    model=progan,
                    optimizer=self.disc_optimizer,  # Note: Should save both, simplified here
                    epoch=epoch,
                    loss=avg_gen_loss,
                    model_name=model_name,
                    training_history=self.training_history,
                    config=self.config,
                    is_best=is_best
                )
                
                if is_best:
                    self.logger.info(f"Saved best model checkpoint: {checkpoint_path}")
        
        # Save final model as trained
        final_progan = ProteinProGAN(
            vocab_size=self.config.get('vocab_size', 21),
            latent_dim=self.config.get('latent_dim', 128)
        )
        final_progan.generator = self.generator
        final_progan.discriminator = self.discriminator
        
        model_path = self.model_saver.save_model(
            model=final_progan,
            model_type="progan",
            model_name=f"{model_name}_final",
            config=self.config,
            is_pretrained=False,
            training_history=self.training_history
        )
        
        self.logger.info(f"Training completed. Final model saved: {model_path}")
        return self.training_history
    
    def generate_sequences(self, num_sequences: int = 10) -> List[str]:
        """Generate protein sequences using the trained generator"""
        self.generator.eval()
        
        with torch.no_grad():
            # Generate noise
            noise = torch.randn(num_sequences, self.config.get('latent_dim', 128), device=self.device)
            
            # Generate sequences
            sequences = self.generator(noise, return_sequences=True)
            
            # Convert to amino acid sequences
            dataset = ProteinDataset([])  # For aa_to_idx mapping
            generated_sequences = []
            
            for seq_tensor in sequences:
                sequence = ""
                for idx in seq_tensor:
                    if idx.item() in dataset.idx_to_aa:
                        sequence += dataset.idx_to_aa[idx.item()]
                    else:
                        sequence += "A"  # Default
                generated_sequences.append(sequence)
            
        return generated_sequences

class ProteinVAETrainer:
    """Trainer class for Protein VAE"""
    
    def __init__(self, config: Dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.vae = ProteinVAE(
            vocab_size=config.get('vocab_size', 21),
            embed_dim=config.get('embed_dim', 256),
            max_length=config.get('max_length', 512),
            latent_dim=config.get('latent_dim', 128)
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.vae.parameters(),
            lr=config.get('lr', 0.001)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training history
        self.training_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'epoch': []
        }
        
        # Model saver
        self.model_saver = ModelSaver()
    
    def train_step(self, sequences: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, mean, logvar = self.vae(sequences)
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        sequences_flat = sequences.view(-1)
        
        # Reconstruction loss
        recon_loss = self.criterion(logits_flat, sequences_flat)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / batch_size
        
        # Total loss
        total_loss = recon_loss + 0.01 * kl_loss  # Beta-VAE with beta=0.01
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def train(
        self,
        train_sequences: List[str],
        epochs: int,
        batch_size: int = 32,
        save_every: int = 10,
        model_name: str = "vae_training"
    ) -> Dict[str, float]:
        """Train the VAE model"""
        self.logger.info(f"Starting VAE training for {epochs} epochs")
        
        # Create dataset and dataloader
        dataset = ProteinDataset(train_sequences)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            
            # Training loop
            for batch_idx, sequences in enumerate(dataloader):
                sequences = sequences.to(self.device)
                
                # Train step
                losses = self.train_step(sequences)
                
                epoch_total_loss += losses['total_loss']
                epoch_recon_loss += losses['reconstruction_loss']
                epoch_kl_loss += losses['kl_loss']
            
            # Calculate average losses
            num_batches = len(dataloader)
            avg_total_loss = epoch_total_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            
            # Update history
            self.training_history['total_loss'].append(avg_total_loss)
            self.training_history['reconstruction_loss'].append(avg_recon_loss)
            self.training_history['kl_loss'].append(avg_kl_loss)
            self.training_history['epoch'].append(epoch)
            
            # Log progress
            if epoch % 5 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{epochs} - Total: {avg_total_loss:.4f}, "
                    f"Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}"
                )
            
            # Save checkpoint
            if epoch % save_every == 0 or epoch == epochs - 1:
                is_best = avg_total_loss < best_loss
                if is_best:
                    best_loss = avg_total_loss
                
                # Save checkpoint
                checkpoint_path = self.model_saver.save_checkpoint(
                    model=self.vae,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=avg_total_loss,
                    model_name=model_name,
                    training_history=self.training_history,
                    config=self.config,
                    is_best=is_best
                )
                
                if is_best:
                    self.logger.info(f"Saved best VAE checkpoint: {checkpoint_path}")
        
        # Save final model
        model_path = self.model_saver.save_model(
            model=self.vae,
            model_type="vae",
            model_name=f"{model_name}_final",
            config=self.config,
            is_pretrained=False,
            training_history=self.training_history
        )
        
        self.logger.info(f"VAE training completed. Final model saved: {model_path}")
        return self.training_history
    
    def generate_sequences(self, num_sequences: int = 10) -> List[str]:
        """Generate protein sequences using the trained VAE"""
        self.vae.eval()
        
        with torch.no_grad():
            # Generate sequences
            sequences = self.vae.generate(num_sequences, self.device)
            
            # Convert to amino acid sequences
            dataset = ProteinDataset([])  # For aa_to_idx mapping
            generated_sequences = []
            
            for seq_tensor in sequences:
                sequence = ""
                for idx in seq_tensor:
                    if idx.item() in dataset.idx_to_aa:
                        sequence += dataset.idx_to_aa[idx.item()]
                    else:
                        sequence += "A"  # Default
                generated_sequences.append(sequence)
            
        return generated_sequences

def train_progan_model(
    dataset_path: str,
    epochs: int = 100,
    batch_size: int = 32,
    model_name: str = "progan_her2"
) -> Dict[str, any]:
    """Train ProGAN model on protein sequences"""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get ProGAN-specific config
    progan_config = config.get('models', {}).get('sequence_generator', {})
    model_config = {
        'latent_dim': progan_config.get('gan_latent_dim', 128),
        'embed_dim': progan_config.get('gan_hidden_dim', 256),
        'vocab_size': 21,
        'max_length': progan_config.get('max_length', 512),
        'gen_lr': 0.0002,
        'disc_lr': 0.0002
    }
    
    # Load training sequences
    preprocessor = ProteinPreprocessor()
    sequences = preprocessor.load_dataset(dataset_path)
    sequence_texts = [seq['sequence'] for seq in sequences[:1000]]  # Limit for demo
    
    # Initialize trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ProteinProGANTrainer(model_config, device=str(device))
    
    # Train model
    training_history = trainer.train(
        train_sequences=sequence_texts,
        epochs=epochs,
        batch_size=batch_size,
        model_name=model_name
    )
    
    # Generate sample sequences
    generated = trainer.generate_sequences(num_sequences=5)
    
    return {
        'training_history': training_history,
        'generated_sequences': generated,
        'model_config': model_config
    }

def train_vae_model(
    dataset_path: str,
    epochs: int = 100,
    batch_size: int = 32,
    model_name: str = "vae_her2"
) -> Dict[str, any]:
    """Train VAE model on protein sequences"""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get VAE-specific config
    model_config = {
        'latent_dim': 128,
        'embed_dim': 256,
        'vocab_size': 21,
        'max_length': 512,
        'lr': 0.001
    }
    
    # Load training sequences
    preprocessor = ProteinPreprocessor()
    sequences = preprocessor.load_dataset(dataset_path)
    sequence_texts = [seq['sequence'] for seq in sequences[:1000]]  # Limit for demo
    
    # Initialize trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ProteinVAETrainer(model_config, device=str(device))
    
    # Train model
    training_history = trainer.train(
        train_sequences=sequence_texts,
        epochs=epochs,
        batch_size=batch_size,
        model_name=model_name
    )
    
    # Generate sample sequences
    generated = trainer.generate_sequences(num_sequences=5)
    
    return {
        'training_history': training_history,
        'generated_sequences': generated,
        'model_config': model_config
    }

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train protein design models')
    parser.add_argument('--model-type', choices=['progan', 'vae'], required=True,
                        help='Type of model to train')
    parser.add_argument('--dataset', required=True,
                        help='Path to training dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--model-name', default='protein_model',
                        help='Name for the trained model')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Train model
    if args.model_type == 'progan':
        result = train_progan_model(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_name=args.model_name
        )
    else:  # vae
        result = train_vae_model(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_name=args.model_name
        )
    
    print(f"Training completed for {args.model_type}!")
    print(f"Generated {len(result['generated_sequences'])} sample sequences")
    
    # Save results
    import json
    results_file = f"{args.model_name}_training_results.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()