"""
Simple training script with progress display.
Run this to see live training progress.
"""

import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# Import from the optimized trainer
from train_hargs_optimized import (
    SelfOptimizingTrainer, TrainingConfig, logger
)
from enhanced_hargs_diverse_train import (
    HARGSModelWithMaxDiversity, DiversityHARGSDataset,
    load_or_create_datasets
)


def display_progress_bar(current, total, width=50, prefix="Progress", suffix=""):
    """Display a simple ASCII progress bar."""
    filled = int(width * current // total)
    bar = '█' * filled + '░' * (width - filled)
    percent = 100 * current / total
    print(f'\r{prefix}: |{bar}| {percent:.1f}% {suffix}', end='', flush=True)


def main():
    """Main training with simple progress display."""
    print("=" * 60)
    print("🚀 HARGS Training with Progress Display")
    print("=" * 60)
    
    # Load datasets
    print("\n📦 Loading datasets...")
    train_texts, _ = load_or_create_datasets()
    
    # Configuration
    config = TrainingConfig(
        num_epochs=3,
        batch_size=32,
        learning_rate=2e-4,
        auto_optimize=True,
        lr_adaptation=True,
        early_stopping_patience=10,
        mixed_precision=torch.cuda.is_available(),
        save_every=50
    )
    
    # Device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n💻 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = HARGSModelWithMaxDiversity(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        diffusion_hidden_dim=config.diffusion_hidden_dim,
        diffusion_num_layers=config.diffusion_num_layers,
        device=device
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.diffusion_model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Create dataset
    dataset = DiversityHARGSDataset(
        train_texts,
        model.tokenizer,
        augmentation_prob=0.5
    )
    print(f"   Dataset size: {len(dataset):,} examples")
    
    # Initialize trainer
    trainer = SelfOptimizingTrainer(model, config)
    
    # Setup optimizer first
    trainer.setup_optimizer()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # 0 for simpler progress display
        drop_last=True
    )
    
    num_batches = len(dataloader)
    print(f"   Batches per epoch: {num_batches}")
    print(f"   Total steps: {num_batches * config.num_epochs}")
    
    # Training loop with manual progress
    print("\n" + "=" * 60)
    print("📚 Training Started")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 60)
        
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            # Train step
            loss = trainer.train_step(batch)
            epoch_loss += loss
            trainer.global_step += 1
            
            # Progress display every 10 batches
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                avg_loss = epoch_loss / (batch_idx + 1)
                current_lr = trainer.optimizer.param_groups[0]['lr']
                
                progress_bar = int(40 * (batch_idx + 1) / num_batches)
                bar = '█' * progress_bar + '░' * (40 - progress_bar)
                
                print(f'\r  Batch {batch_idx+1}/{num_batches} |{bar}| '
                      f'Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | Best: {best_loss:.6f}',
                      end='', flush=True)
            
            # Adapt learning rate
            if trainer.global_step % 50 == 0:
                trainer.adapt_learning_rate(loss)
            
            # Save checkpoint
            if trainer.global_step % config.save_every == 0:
                is_best = loss < best_loss
                if is_best:
                    best_loss = loss
                trainer.best_loss = best_loss
                trainer.save_checkpoint(is_best)
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\n\n  ✅ Epoch {epoch+1} complete!")
        print(f"     Average Loss: {avg_epoch_loss:.6f}")
        print(f"     Best Loss: {best_loss:.6f}")
        print(f"     Current LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping check
        if trainer.check_early_stopping(avg_epoch_loss):
            print("\n  ⏹️ Early stopping triggered")
            break
    
    # Final save
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    
    final_path = Path("./hargs_optimized_checkpoints/final_model.pth")
    torch.save({
        'global_step': trainer.global_step,
        'model_state_dict': model.state_dict(),
        'best_loss': best_loss,
        'training_log': trainer.training_log
    }, final_path)
    
    print(f"\n💾 Final model saved: {final_path}")
    print(f"🏆 Best loss achieved: {best_loss:.6f}")
    print(f"📊 Total steps: {trainer.global_step}")
    
    # Quick test
    print("\n🧪 Quick generation test:")
    model.eval()
    test_query = "What is machine learning?"
    
    for i in range(3):
        with torch.no_grad():
            result = model(test_query, temperature=2.0)
            print(f"  {i+1}. {result['response'][:70]}...")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
