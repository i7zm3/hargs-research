"""
FineWeb EDU HARGS Training Script

This script implements fast training using the HuggingFaceFW/fineweb-edu dataset (1.3T tokens)
with streaming to achieve 0.7 loss as quickly as possible.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
import time
import json
import os
from pathlib import Path
import argparse
import hashlib
try:
    import wandb  # For experiment tracking # type: ignore
except ImportError:
    wandb = None
    print("Warning: wandb not available, skipping experiment tracking")

try:
    import sklearn.metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    print("Warning: sklearn not available, some metrics will be skipped")
    # Define placeholder functions if sklearn is not available
    def accuracy_score(*args, **kwargs):
        raise NotImplementedError("sklearn not available")
    
    def precision_recall_fscore_support(*args, **kwargs):
        raise NotImplementedError("sklearn not available")

import matplotlib.pyplot as plt
try:
    import seaborn as sns  # type: ignore
except ImportError:
    print("Warning: seaborn not available, skipping advanced plotting")
    sns = None

# Import HARGS model components
from hargs_model import HARGSModel, HARGSDataset, train_hargs_model
from hargs_model import HierarchicalTokenizer, SplitHalfDiffusion, UsageWeightCalculator

# Import enhanced diversity components
from enhanced_hargs_diverse_train import HARGSModelWithMaxDiversity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FineWebDataset(Dataset):
    """Dataset wrapper for FineWeb EDU dataset."""
    
    def __init__(self, dataset_stream, tokenizer, max_length=512, num_samples=10000):
        self.dataset_stream = dataset_stream
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = min(num_samples, 100000)  # Limit for memory management
        
        # Pre-load a limited number of samples for faster access
        self.samples = []
        count = 0
        for item in dataset_stream:
            if count >= self.num_samples:
                break
            text = item.get('text', '') or item.get('content', '') or ''
            if text and len(text) > 10:  # Filter out very short texts
                self.samples.append(text)
                count += 1
            if count % 1000 == 0:
                logger.info(f"Loaded {count} samples from FineWeb EDU...")
        
        logger.info(f"Loaded {len(self.samples)} samples for training")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Create input and target embeddings
        tokens = self.tokenizer.tokenize(text[:1000])  # Limit text length
        
        if tokens:
            # Use first token's embedding as representative
            input_emb = torch.from_numpy(tokens[0].embedding).float()
            
            # Create target embedding (next token or variation)
            if len(tokens) > 1:
                target_emb = torch.from_numpy(tokens[1].embedding).float()
            else:
                # If only one token, create a slightly modified version
                target_emb = input_emb + 0.1 * torch.randn_like(input_emb)
        else:
            # Fallback if no tokens
            input_emb = torch.zeros(self.tokenizer.embedding_dim)
            target_emb = torch.zeros(self.tokenizer.embedding_dim)
        
        return input_emb, target_emb


class FastFineWebTrainer:
    """Fast trainer using FineWeb EDU dataset with streaming."""
    
    def __init__(self, 
                 model: HARGSModelWithMaxDiversity,
                 dataset_stream,
                 output_dir: str = "./fineweb_hargs_checkpoints",
                 use_wandb: bool = False):
        self.model = model
        self.dataset_stream = dataset_stream
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_wandb = use_wandb
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        if self.use_wandb:
            wandb.init(project="hargs-fineweb-fast-training", config={
                "model_type": "HARGS-FineWeb",
                "vocab_size": model.tokenizer.vocab_size,
                "embedding_dim": model.diffusion_model.embed_dim,
                "diffusion_layers": model.diffusion_model.num_layers,
                "dataset": "HuggingFaceFW/fineweb-edu",
                "streaming": True,
                "fast_training": True
            })
    
    def setup_optimizer(self, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        """Setup optimizer with higher learning rate for faster convergence."""
        # Use different learning rates for different components
        param_groups = [
            {
                'params': list(self.model.diffusion_model.parameters()),
                'lr': learning_rate,
                'name': 'diffusion'
            },
            {
                'params': list(self.model.tokenizer.parameters()) if hasattr(self.model.tokenizer, 'parameters') else [],
                'lr': learning_rate * 0.1,  # Lower LR for tokenizer
                'name': 'tokenizer'
            }
        ]
        
        # Filter out empty parameter groups
        param_groups = [pg for pg in param_groups if len(pg['params']) > 0]
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler for fast convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,  # Moderate reduction
            patience=2   # Quick adjustment for fast training
        )
    
    def train_batch(self, input_embs, target_embs) -> float:
        """Train a single batch."""
        input_embs = input_embs.to(self.model.device)
        target_embs = target_embs.to(self.model.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass through diffusion model
        pred_noise, true_noise = self.model.diffusion_model(target_embs, input_embs)
        
        # Calculate loss
        loss = self.criterion(pred_noise, true_noise)
        
        # Add diversity regularization (light for speed)
        diversity_reg = 0.01 * torch.std(pred_noise)  # Mild diversity encouragement
        total_loss_with_reg = loss + diversity_reg
        
        # Backward pass
        total_loss_with_reg.backward()
        
        # Gradient clipping (for stability with large dataset)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate_batch(self, input_embs, target_embs) -> float:
        """Validate a single batch."""
        input_embs = input_embs.to(self.model.device)
        target_embs = target_embs.to(self.model.device)
        
        with torch.no_grad():
            # Forward pass
            pred_noise, true_noise = self.model.diffusion_model(target_embs, input_embs)
            
            # Calculate loss
            loss = self.criterion(pred_noise, true_noise)
        
        return loss.item()
    
    def train(self, 
              num_epochs: int = 1,  # Minimal epochs for fast training
              batch_size: int = 64,  # Reasonable batch size for memory
              learning_rate: float = 1e-3,  # Higher LR for faster convergence
              max_samples: int = 100000,  # Limit samples for practical training
              target_loss: float = 0.7,  # Target loss
              save_every: int = 500,  # Save every 500 steps
              checkpoint_interval: int = 10000) -> Dict[str, Any]:
        """Main training loop with streaming dataset."""
        logger.info(f"Starting FAST FineWeb training with target loss: {target_loss}...")
        
        # Setup optimizer
        self.setup_optimizer(learning_rate)
        
        # Create dataset with limited samples for practical training
        fineweb_dataset = FineWebDataset(self.dataset_stream, self.model.tokenizer, num_samples=max_samples)
        
        # Create data loader
        dataloader = DataLoader(
            fineweb_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,  # Reduced for streaming
            pin_memory=False  # Disabled for streaming
        )
        
        best_val_loss = float('inf')
        training_history = []
        step_count = 0
        total_loss = 0.0
        
        logger.info(f"Starting fast training with {len(fineweb_dataset)} samples...")
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (input_embs, target_embs) in enumerate(dataloader):
                # Train batch
                batch_loss = self.train_batch(input_embs, target_embs)
                
                epoch_loss += batch_loss
                total_loss += batch_loss
                num_batches += 1
                step_count += 1
                
                # Update learning rate scheduler periodically
                if step_count % 100 == 0:
                    self.scheduler.step(batch_loss)
                
                # Log progress
                if batch_idx % 100 == 0:
                    avg_loss = total_loss / max(step_count, 1)
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Step {step_count}, "
                               f"Batch Loss: {batch_loss:.6f}, Avg Loss: {avg_loss:.6f}")
                
                # Check if we've reached target loss
                if avg_loss <= target_loss:
                    logger.info(f"Target loss {target_loss} reached at step {step_count}! Avg Loss: {avg_loss:.6f}")
                    break
                
                # Save checkpoint periodically
                if step_count % save_every == 0:
                    self.save_checkpoint(epoch, step_count, batch_loss, avg_loss, is_best=True)
                
                # Stop if we've reached target loss
                if avg_loss <= target_loss:
                    break
            
            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            self.train_losses.append(avg_epoch_loss)
            
            logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.6f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_epoch_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'steps_completed': step_count,
                    'cumulative_avg_loss': total_loss / max(step_count, 1)
                })
            
            # Check if target loss is achieved
            if avg_epoch_loss <= target_loss:
                logger.info(f"Target loss {target_loss} achieved! Stopping training.")
                break
        
        logger.info("Fast FineWeb training completed!")
        
        # Final metrics
        final_avg_loss = total_loss / max(step_count, 1)
        
        return {
            'training_history': training_history,
            'final_avg_loss': final_avg_loss,
            'steps_completed': step_count,
            'target_loss_achieved': final_avg_loss <= target_loss
        }
    
    def save_checkpoint(self, epoch: int, step: int, batch_loss: float, avg_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'batch_loss': batch_loss,
            'avg_loss': avg_loss,
            'model_config': {
                'vocab_size': self.model.tokenizer.vocab_size,
                'embedding_dim': self.model.diffusion_model.embed_dim,
                'diffusion_hidden_dim': self.model.diffusion_model.hidden_dim,
                'diffusion_num_layers': self.model.diffusion_model.num_layers,
                'device': str(self.model.device)
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"fineweb_checkpoint_step_{step}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_fineweb_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best FineWeb model with avg_loss: {avg_loss:.6f}")


def main():
    """Main training function with FineWeb EDU dataset."""
    parser = argparse.ArgumentParser(description="Train HARGS Model with FineWeb EDU Dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_samples", type=int, default=100000, help="Maximum samples to use")
    parser.add_argument("--target_loss", type=float, default=0.7, help="Target loss to achieve")
    parser.add_argument("--output_dir", type=str, default="./fineweb_hargs_checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Load FineWeb EDU dataset with streaming
    logger.info("Loading FineWeb EDU dataset with streaming...")
    try:
        dataset_stream = load_dataset("HuggingFaceFW/fineweb-edu", streaming=True, split="train")
        logger.info("Dataset loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Make sure you have datasets library installed: pip install datasets")
        return
    
    # Initialize model with maximum diversity enhancements
    logger.info(f"Initializing HARGS model with diversity enhancements...")
    model = HARGSModelWithMaxDiversity(
        vocab_size=10000,
        embedding_dim=512,
        diffusion_hidden_dim=1024,
        diffusion_num_layers=6
    )
    
    # Initialize trainer
    trainer = FastFineWebTrainer(
        model=model,
        dataset_stream=dataset_stream,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )
    
    # Start training with fast parameters
    logger.info("Starting FAST FineWeb training process...")
    logger.info(f"Target: {args.target_loss} loss")
    start_time = time.time()
    
    results = trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        target_loss=args.target_loss,
        save_every=500,  # Save every 500 steps
        checkpoint_interval=10000  # Checkpoint interval
    )
    
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Print final results
    logger.info("FineWeb training completed!")
    logger.info(f"Final average loss: {results['final_avg_loss']:.6f}")
    logger.info(f"Steps completed: {results['steps_completed']}")
    logger.info(f"Target loss achieved: {results['target_loss_achieved']}")
    logger.info(f"Training duration: {training_duration:.2f}s ({training_duration/60:.2f} minutes)")
    
    # Save training results
    results_path = Path(args.output_dir) / "fineweb_training_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'final_avg_loss': results['final_avg_loss'],
            'steps_completed': results['steps_completed'],
            'target_loss_achieved': results['target_loss_achieved'],
            'training_duration': training_duration,
            'target_loss_set': args.target_loss,
            'batch_size_used': args.batch_size,
            'learning_rate_used': args.learning_rate
        }, f, indent=2)
    
    logger.info(f"Training results saved to {results_path}")
    
    # Test the model briefly
    logger.info("Testing trained model...")
    test_queries = [
        "What is machine learning?",
        "Explain quantum computing briefly",
        "How does photosynthesis work?"
    ]
    
    for query in test_queries:
        result = model(query)
        print(f"\nQuery: {query}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Strategy: {result['strategy_used']}")
        print(f"Latency: {result['latency']:.3f}s")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
