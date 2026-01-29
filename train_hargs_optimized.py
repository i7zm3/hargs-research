"""
Optimized HARGS Training with Progress Display and Self-Optimization
Features:
- Real-time progress bars with tqdm
- Self-optimizing hyperparameters (learning rate, batch size)
- Live metrics dashboard
- Automatic checkpointing
- Early stopping with patience
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import random
from collections import defaultdict, deque
import sys

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    from tqdm.auto import trange
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available, using simple progress")

# Import HARGS components
from enhanced_hargs_diverse_train import (
    HARGSModelWithMaxDiversity, DiversityHARGSDataset,
    MaxDiversitySplitHalfDiffusion, load_or_create_datasets
)

# Configure logging with colors
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'DEBUG': '\033[94m',    # Blue
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class TrainingConfig:
    """Training configuration with self-optimization."""
    # Model params
    vocab_size: int = 10000
    embedding_dim: int = 512
    diffusion_hidden_dim: int = 1024
    diffusion_num_layers: int = 6
    
    # Training params (will be auto-optimized)
    learning_rate: float = 2e-4
    batch_size: int = 32
    num_epochs: int = 5
    weight_decay: float = 1e-4
    
    # Self-optimization params
    auto_optimize: bool = True
    lr_adaptation: bool = True
    batch_size_adaptation: bool = True
    gradient_accumulation: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    min_delta: float = 0.001
    
    # Checkpointing
    save_every: int = 100
    max_checkpoints: int = 3
    
    # Performance
    mixed_precision: bool = True
    num_workers: int = 2
    pin_memory: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LiveMetricsTracker:
    """Tracks and displays live training metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)
        self.start_time = time.time()
        self.step_count = 0
        
    def update(self, loss: float, lr: float, step_time: float):
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.step_times.append(step_time)
        self.step_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.losses:
            return {}
        
        return {
            'avg_loss': np.mean(self.losses),
            'std_loss': np.std(self.losses),
            'min_loss': min(self.losses),
            'max_loss': max(self.losses),
            'avg_step_time': np.mean(self.step_times),
            'steps_per_sec': 1.0 / np.mean(self.step_times) if self.step_times else 0,
            'current_lr': self.learning_rates[-1] if self.learning_rates else 0,
            'elapsed_time': time.time() - self.start_time,
            'total_steps': self.step_count
        }
    
    def format_progress(self) -> str:
        stats = self.get_stats()
        if not stats:
            return "Initializing..."
        
        return (
            f"Loss: {stats['avg_loss']:.4f}±{stats['std_loss']:.4f} | "
            f"LR: {stats['current_lr']:.2e} | "
            f"Speed: {stats['steps_per_sec']:.1f} steps/s | "
            f"Time: {stats['elapsed_time']:.1f}s"
        )


class SelfOptimizingTrainer:
    """Trainer with self-optimization capabilities."""
    
    def __init__(self, model: HARGSModelWithMaxDiversity, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = model.device
        self.output_dir = Path("./hargs_optimized_checkpoints")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Training state
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
        
        # Metrics tracking
        self.metrics = LiveMetricsTracker()
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        
        # Adaptive state
        self.loss_history = deque(maxlen=50)
        self.lr_history = deque(maxlen=10)
        self.effective_batch_size = config.batch_size
        self.accumulation_steps = 1
        
        # Training log
        self.training_log = []
        
        logger.info(f"🚀 SelfOptimizingTrainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Mixed Precision: {self.scaler is not None}")
        logger.info(f"   Auto-optimize: {config.auto_optimize}")
    
    def setup_optimizer(self):
        """Setup optimizer with current config."""
        params = list(self.model.diffusion_model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8
        )
        
        logger.info(f"📊 Optimizer: lr={self.config.learning_rate:.2e}, wd={self.config.weight_decay}")
    
    def adapt_learning_rate(self, current_loss: float):
        """Self-adapt learning rate based on training dynamics."""
        if not self.config.lr_adaptation:
            return
        
        self.loss_history.append(current_loss)
        if len(self.loss_history) < 10:
            return
        
        # Calculate loss trend
        recent_avg = np.mean(list(self.loss_history)[-5:])
        older_avg = np.mean(list(self.loss_history)[:5])
        
        # If loss not improving, reduce LR
        if recent_avg > older_avg * 0.99:  # Less than 1% improvement
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * 0.95
                param_group['lr'] = new_lr
                self.lr_history.append(new_lr)
                logger.info(f"📉 LR adapted: {old_lr:.2e} → {new_lr:.2e}")
                break
        
        # If improving fast, increase LR slightly
        elif recent_avg < older_avg * 0.9:  # More than 10% improvement
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                if old_lr < 1e-3:  # Cap at 1e-3
                    new_lr = old_lr * 1.02
                    param_group['lr'] = new_lr
                    self.lr_history.append(new_lr)
                    logger.info(f"📈 LR increased: {old_lr:.2e} → {new_lr:.2e}")
                    break
    
    def adapt_batch_size(self, dataloader: DataLoader) -> DataLoader:
        """Adapt batch size based on memory and performance."""
        if not self.config.batch_size_adaptation:
            return dataloader
        
        # Check if we can increase batch size
        if len(self.loss_history) > 20:
            recent_losses = list(self.loss_history)[-20:]
            loss_variance = np.var(recent_losses)
            
            # If training is stable, try larger batch
            if loss_variance < 0.01 and self.effective_batch_size < 128:
                self.effective_batch_size = min(self.effective_batch_size + 8, 128)
                logger.info(f"🔧 Batch size increased to {self.effective_batch_size}")
                
                # Recreate dataloader
                dataset = dataloader.dataset
                return DataLoader(
                    dataset,
                    batch_size=self.effective_batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    drop_last=True
                )
        
        return dataloader
    
    def diversity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute diversity encouragement loss."""
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Normalize
        embeddings_norm = F.normalize(embeddings, dim=-1)
        
        # Pairwise cosine similarity
        similarity = torch.matmul(embeddings_norm, embeddings_norm.t())
        
        # Mask diagonal
        mask = ~torch.eye(batch_size, device=embeddings.device).bool()
        
        # Negative mean similarity = diversity
        diversity = -similarity[mask].mean()
        
        return diversity
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], accumulation_step: int = 0) -> float:
        """Single training step with optional gradient accumulation."""
        step_start = time.time()
        
        input_emb, target_emb = batch
        input_emb = input_emb.to(self.device)
        target_emb = target_emb.to(self.device)
        
        # Mixed precision forward
        if self.scaler:
            with torch.cuda.amp.autocast():
                pred_noise, true_noise = self.model.diffusion_model(target_emb, input_emb)
                mse_loss = self.criterion(pred_noise, true_noise)
                div_loss = 0.01 * self.diversity_loss(pred_noise)
                loss = mse_loss + div_loss
        else:
            pred_noise, true_noise = self.model.diffusion_model(target_emb, input_emb)
            mse_loss = self.criterion(pred_noise, true_noise)
            div_loss = 0.01 * self.diversity_loss(pred_noise)
            loss = mse_loss + div_loss
        
        # Scale loss for accumulation
        if self.config.gradient_accumulation:
            loss = loss / self.accumulation_steps
        
        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights only on final accumulation step
        if not self.config.gradient_accumulation or accumulation_step == self.accumulation_steps - 1:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        step_time = time.time() - step_start
        loss_value = loss.item() * (self.accumulation_steps if self.config.gradient_accumulation else 1)
        
        # Update metrics
        current_lr = self.optimizer.param_groups[0]['lr']
        self.metrics.update(loss_value, current_lr, step_time)
        
        return loss_value
    
    def check_early_stopping(self, current_loss: float) -> bool:
        """Check if training should stop early."""
        if current_loss < self.best_loss - self.config.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"⏹️ Early stopping triggered after {self.patience_counter} steps without improvement")
                return True
        return False
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.to_dict(),
            'training_log': self.training_log
        }
        
        # Regular checkpoint
        if self.global_step % self.config.save_every == 0:
            path = self.output_dir / f"checkpoint_step_{self.global_step}.pth"
            torch.save(checkpoint, path)
        
        # Best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"💾 Best model saved (loss: {self.best_loss:.6f})")
    
    def display_progress(self, epoch: int, num_epochs: int, batch_idx: int, num_batches: int):
        """Display training progress."""
        if TQDM_AVAILABLE:
            # progress bar is handled by tqdm
            return
        
        # Simple console output
        progress = (batch_idx + 1) / num_batches * 100
        stats = self.metrics.format_progress()
        logger.info(f"Epoch {epoch+1}/{num_epochs} [{progress:.1f}%] | {stats}")
    
    def train(self, dataset: DiversityHARGSDataset) -> Dict[str, Any]:
        """Main training loop with self-optimization."""
        logger.info("\n" + "="*60)
        logger.info("🚀 Starting Self-Optimizing Training")
        logger.info("="*60)
        
        # Setup
        self.setup_optimizer()
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        num_batches = len(dataloader)
        logger.info(f"📦 Dataset: {len(dataset)} samples, {num_batches} batches")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n📚 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Optionally adapt batch size
            if epoch > 0 and epoch % 2 == 0:
                dataloader = self.adapt_batch_size(dataloader)
                num_batches = len(dataloader)
            
            # Progress bar
            if TQDM_AVAILABLE:
                pbar = tqdm(enumerate(dataloader), total=num_batches, desc=f"Epoch {epoch+1}")
            else:
                pbar = enumerate(dataloader)
            
            epoch_loss = 0.0
            num_steps = 0
            
            self.model.train()
            for batch_idx, batch in pbar:
                # Gradient accumulation
                for accum_step in range(self.accumulation_steps):
                    loss = self.train_step(batch, accum_step)
                    epoch_loss += loss
                    num_steps += 1
                    self.global_step += 1
                
                # Adapt learning rate
                if self.global_step % 50 == 0:
                    self.adapt_learning_rate(loss)
                
                # Update progress bar
                if TQDM_AVAILABLE:
                    stats = self.metrics.get_stats()
                    pbar.set_postfix({
                        'loss': f"{stats.get('avg_loss', 0):.4f}",
                        'lr': f"{stats.get('current_lr', 0):.2e}",
                        'best': f"{self.best_loss:.4f}"
                    })
                elif batch_idx % 10 == 0:
                    self.display_progress(epoch, self.config.num_epochs, batch_idx, num_batches)
                
                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    is_best = loss < self.best_loss
                    if is_best:
                        self.best_loss = loss
                    self.save_checkpoint(is_best)
                
                # Check early stopping
                if self.global_step % 20 == 0 and self.check_early_stopping(loss):
                    break
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / max(num_steps, 1)
            stats = self.metrics.get_stats()
            
            epoch_log = {
                'epoch': epoch + 1,
                'avg_loss': avg_epoch_loss,
                'best_loss': self.best_loss,
                'steps': self.global_step,
                'learning_rate': stats.get('current_lr', 0),
                'time_elapsed': stats.get('elapsed_time', 0)
            }
            self.training_log.append(epoch_log)
            
            logger.info(f"✅ Epoch {epoch+1} complete: avg_loss={avg_epoch_loss:.6f}, best={self.best_loss:.6f}")
        
        # Final save
        final_path = self.output_dir / "final_model.pth"
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_loss': self.best_loss,
            'training_log': self.training_log,
            'config': self.config.to_dict()
        }, final_path)
        
        logger.info(f"\n🎉 Training complete!")
        logger.info(f"   Best loss: {self.best_loss:.6f}")
        logger.info(f"   Total steps: {self.global_step}")
        logger.info(f"   Final model: {final_path}")
        
        return {
            'best_loss': self.best_loss,
            'training_log': self.training_log,
            'final_model_path': str(final_path),
            'total_steps': self.global_step
        }


def main():
    """Main entry point."""
    # Load datasets
    logger.info("Loading datasets...")
    train_texts, _ = load_or_create_datasets()
    
    # Configuration
    config = TrainingConfig(
        num_epochs=5,
        batch_size=32,
        learning_rate=2e-4,
        auto_optimize=True,
        lr_adaptation=True,
        early_stopping_patience=10,
        mixed_precision=torch.cuda.is_available()
    )
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = HARGSModelWithMaxDiversity(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        diffusion_hidden_dim=config.diffusion_hidden_dim,
        diffusion_num_layers=config.diffusion_num_layers,
        device=device
    )
    
    # Create dataset
    dataset = DiversityHARGSDataset(
        train_texts,
        model.tokenizer,
        augmentation_prob=0.5
    )
    
    # Initialize trainer
    trainer = SelfOptimizingTrainer(model, config)
    
    # Train
    results = trainer.train(dataset)
    
    # Save results
    results_path = Path("./hargs_optimized_checkpoints/training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    
    # Test generation
    logger.info("\n🧪 Testing generation diversity...")
    model.eval()
    test_query = "What is machine learning?"
    
    with torch.no_grad():
        for i in range(5):
            result = model(test_query, temperature=2.0)
            logger.info(f"  {i+1}. {result['response'][:80]}...")
    
    logger.info("\n✨ All done!")


if __name__ == "__main__":
    main()
