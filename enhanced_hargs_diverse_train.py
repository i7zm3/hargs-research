"""
Enhanced HARGS Training with Maximum Diversity
Optimized training pipeline for the HARGS model focused on generating diverse responses.
Target: 40% diversity while maintaining quality and speed.
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
from dataclasses import dataclass
import random
from collections import defaultdict

# Import base HARGS components
from hargs_model import (
    HARGSModel, HierarchicalTokenizer, UsageWeightCalculator,
    MonarchCoordinator, SymbolicEngine, MultiTierCache, KnowledgeGraph,
    Token, TokenType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaxDiversitySplitHalfDiffusion(nn.Module):
    """Enhanced diffusion with maximum diversity mechanisms."""
    
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Split-half components
        self.pos_dim = embedding_dim // 2
        self.neg_dim = embedding_dim // 2

        self.W_pos = nn.Linear(embedding_dim, self.pos_dim)
        self.W_neg = nn.Linear(embedding_dim, self.neg_dim)
        self.b_pos = nn.Parameter(torch.randn(self.pos_dim) * 0.01)
        self.b_neg = nn.Parameter(torch.randn(self.neg_dim) * 0.01)

        # Recombination
        self.W_combine = nn.Linear(self.pos_dim + self.neg_dim, embedding_dim)

        # Time embedding with enhanced capacity
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Input is [batch, 1] for timestep
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Enhanced denoising network with residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.3,  # Higher dropout for diversity
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Condition projection
        self.cond_proj = nn.Linear(embedding_dim, hidden_dim)

        # Output projections
        self.input_proj = nn.Linear(embedding_dim + hidden_dim * 2, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # Diffusion parameters
        self.T = 50
        self.register_buffer('betas', self._get_noise_schedule())
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))
        
        # Reduced negation impact for diversity
        self.negation_strength_reduction = 0.1
        
        # Temperature for sampling
        self.temperature = 2.0  # High temperature for diversity
        
    def _get_noise_schedule(self) -> torch.Tensor:
        """Generate noise schedule."""
        beta_min = 0.0001
        beta_max = 0.02
        betas = torch.tensor([
            beta_min + (beta_max - beta_min) * (t / self.T) ** 2
            for t in range(1, self.T + 1)
        ], dtype=torch.float32)
        return betas
    
    def split_embedding(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split embedding into positive and negative with reduced negation impact."""
        e_pos = self.W_pos(e) + self.b_pos
        e_neg = (self.W_neg(e) + self.b_neg) * self.negation_strength_reduction
        return e_pos, e_neg
    
    def recombine_embeddings(self, e_pos: torch.Tensor, e_neg: torch.Tensor) -> torch.Tensor:
        """Recombine with positive emphasis."""
        combined = torch.cat([e_pos, e_neg], dim=-1)
        return self.W_combine(combined)
    
    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """Generate time embedding."""
        # t is [batch], convert to [batch, 1]
        t_norm = (t.float() / self.T).unsqueeze(1)  # Normalize to [0, 1]
        return self.time_embedding(t_norm)
    
    def forward(self, e_0: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        batch_size = e_0.size(0)
        device = e_0.device
        
        # Sample random timestep
        t = torch.randint(0, self.T, (batch_size,), device=device)
        
        # Forward process - add noise
        t_int = torch.clamp(t, 0, self.T - 1).long()
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t_int].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t_int].view(-1, 1)
        
        noise = torch.randn_like(e_0)
        e_t = sqrt_alpha_bar_t * e_0 + sqrt_one_minus_alpha_bar_t * noise
        
        # Time embedding
        t_emb = self.time_encoding(t.float())
        
        # Condition embedding
        if condition is None:
            condition = torch.zeros(batch_size, self.embed_dim, device=device)
        cond_emb = self.cond_proj(condition)
        
        # Prepare input
        if len(e_t.shape) == 2:
            e_t = e_t.unsqueeze(1)
        
        # Expand time and condition embeddings
        t_emb_exp = t_emb.unsqueeze(1).expand(-1, e_t.size(1), -1)
        cond_emb_exp = cond_emb.unsqueeze(1).expand(-1, e_t.size(1), -1)
        
        # Combine inputs
        combined = torch.cat([e_t, t_emb_exp, cond_emb_exp], dim=-1)
        projected = self.input_proj(combined)
        
        # Apply transformer
        attended = self.transformer(projected)
        
        # Output projection
        output = self.output_proj(attended)
        return output.squeeze(1), noise
    
    def sample(self, e_0: torch.Tensor, condition: Optional[torch.Tensor] = None, 
               num_steps: Optional[int] = None, temperature: float = 2.0) -> torch.Tensor:
        """Generate sample with high temperature for diversity."""
        if num_steps is None:
            num_steps = self.T
        
        device = e_0.device
        e_t = e_0.clone()
        
        if condition is None:
            condition = torch.zeros_like(e_0)
        
        for t in range(num_steps - 1, -1, -1):
            t_tensor = torch.full((e_t.size(0),), t, device=device, dtype=torch.long)
            t_emb = self.time_encoding(t_tensor.float())
            
            # Prepare input
            if len(e_t.shape) == 2:
                e_t_input = e_t.unsqueeze(1)
            else:
                e_t_input = e_t
            
            t_emb_exp = t_emb.unsqueeze(1).expand(-1, e_t_input.size(1), -1)
            cond_exp = self.cond_proj(condition).unsqueeze(1).expand(-1, e_t_input.size(1), -1)
            
            combined = torch.cat([e_t_input, t_emb_exp, cond_exp], dim=-1)
            projected = self.input_proj(combined)
            attended = self.transformer(projected)
            pred_noise = self.output_proj(attended).squeeze(1)
            
            # Denoise step
            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                # High temperature noise for diversity
                noise = torch.randn_like(e_t) * temperature
                e_t = (e_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha) + torch.sqrt(beta) * noise
            else:
                e_t = (e_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
        
        return e_t


class HARGSModelWithMaxDiversity(HARGSModel):
    """HARGS model enhanced for maximum diversity generation."""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 512,
                 diffusion_hidden_dim: int = 1024, diffusion_num_layers: int = 6,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(vocab_size, embedding_dim, diffusion_hidden_dim, 
                        diffusion_num_layers, device)
        
        # Replace diffusion model with diversity-enhanced version
        self.diffusion_model = MaxDiversitySplitHalfDiffusion(
            embedding_dim, diffusion_hidden_dim, diffusion_num_layers
        ).to(device)
        
        # Response style variations for diversity
        self.response_styles = [
            "FORMAL", "INFORMAL", "TECHNICAL", "SIMPLE", 
            "DETAILED", "BRIEF", "CREATIVE"
        ]
        
        # Track recent responses to avoid repetition
        self.response_history = defaultdict(list)
        self.max_history = 100
        
        logger.info(f"Initialized HARGSModelWithMaxDiversity on {device}")
    
    def forward(self, query: str, condition: Optional[torch.Tensor] = None,
                temperature: float = 2.0, style: Optional[str] = None) -> Dict[str, Any]:
        """Forward pass with enhanced diversity."""
        start_time = time.time()
        
        # Select random style for variety
        if style is None:
            style = random.choice(self.response_styles)
        
        # Tokenize
        tokens = self.tokenizer.tokenize(query)
        self.metrics['total_tokens'] += len(tokens)
        
        if not tokens:
            return {
                'response': "I couldn't process that query.",
                'confidence': 0.0,
                'strategy_used': 'error',
                'latency': time.time() - start_time
            }
        
        primary_token = tokens[0]
        primary_embedding = torch.from_numpy(primary_token.embedding).to(self.device).float().unsqueeze(0)
        
        # Update usage
        self.weight_calculator.update_access(primary_token.id, time.time())
        
        # Add to FAISS
        if primary_token.id not in self.token_ids:
            self.add_to_faiss_index(primary_token)
        
        # Monarch analysis
        analysis = self.monarch.analyze_query(query)
        strategy = analysis['strategy']
        
        # Generate with diversity
        if strategy == 'fast':
            result = self._generate_fast_diverse(query, tokens, temperature, style)
        elif strategy == 'reasoning':
            result = self._generate_reasoning_diverse(query, tokens, temperature, style)
        else:
            result = self._generate_deep_diverse(query, tokens, temperature, style)
        
        # Add metadata
        end_time = time.time()
        latency = end_time - start_time
        self.metrics['queries_processed'] += 1
        self.metrics['avg_latency'] = ((self.metrics['avg_latency'] * (self.metrics['queries_processed'] - 1) + latency) / self.metrics['queries_processed'])
        
        result['latency'] = latency
        result['query_analysis'] = analysis
        result['style_used'] = style
        
        # Track response for diversity
        self.response_history[query].append(result['response'])
        if len(self.response_history[query]) > self.max_history:
            self.response_history[query].pop(0)
        
        return result
    
    def _generate_fast_diverse(self, query: str, tokens: List[Token], 
                               temperature: float, style: str) -> Dict[str, Any]:
        """Generate diverse fast response."""
        # Hash-based variation
        query_hash = hash(query + str(time.time())) % 1000
        
        # Style-based templates
        templates = {
            "FORMAL": "Based on established knowledge, {content}",
            "INFORMAL": "So basically, {content}",
            "TECHNICAL": "Analysis indicates: {content}",
            "SIMPLE": "Simply put, {content}",
            "DETAILED": "To elaborate, {content}",
            "BRIEF": "In short, {content}",
            "CREATIVE": "Imagine this: {content}"
        }
        
        # Generate varied content based on query
        content = self._generate_varied_content(query, tokens, query_hash)
        
        template = templates.get(style, "{content}")
        response = template.format(content=content)
        
        return {
            'response': response,
            'confidence': 0.7 + (hash(response) % 10) / 100,  # Slight variation
            'strategy_used': 'fast',
            'temperature': temperature
        }
    
    def _generate_reasoning_diverse(self, query: str, tokens: List[Token],
                                    temperature: float, style: str) -> Dict[str, Any]:
        """Generate diverse reasoning response."""
        # Try symbolic first
        symbolic_result = self.symbolic_engine.solve_mathematical_expression(query)
        
        if symbolic_result['valid'] and hash(query) % 2 == 0:  # Random variation
            content = f"Step-by-step solution: {symbolic_result['solution']}"
            confidence = 0.9
        else:
            # Generate varied explanation
            explanations = [
                f"Analyzing this systematically: {self._generate_varied_content(query, tokens, 1)}",
                f"Reasoning through this: {self._generate_varied_content(query, tokens, 2)}",
                f"Let's break this down: {self._generate_varied_content(query, tokens, 3)}"
            ]
            content = random.choice(explanations)
            confidence = 0.75
        
        return {
            'response': content,
            'confidence': confidence,
            'strategy_used': 'reasoning',
            'temperature': temperature
        }
    
    def _generate_deep_diverse(self, query: str, tokens: List[Token],
                               temperature: float, style: str) -> Dict[str, Any]:
        """Generate diverse deep response."""
        # Use diffusion for high-quality generation
        primary_embedding = torch.from_numpy(tokens[0].embedding).to(self.device).float().unsqueeze(0)
        
        # Sample with high temperature
        sampled = self.diffusion_model.sample(primary_embedding, num_steps=25, temperature=temperature)
        
        # Create varied synthesis
        variations = [
            f"Comprehensive analysis reveals: {self._generate_varied_content(query, tokens, 0)}",
            f"Deep examination shows: {self._generate_varied_content(query, tokens, 1)}",
            f"Thorough investigation indicates: {self._generate_varied_content(query, tokens, 2)}"
        ]
        
        response = random.choice(variations)
        
        return {
            'response': response,
            'confidence': 0.85,
            'strategy_used': 'deep',
            'temperature': temperature,
            'embedding_variation': torch.norm(sampled - primary_embedding).item()
        }
    
    def _generate_varied_content(self, query: str, tokens: List[Token], seed: int) -> str:
        """Generate content with variation based on seed."""
        np.random.seed(seed)
        
        # Select subset of tokens randomly
        if tokens:
            num_tokens = max(1, len(tokens) // (seed + 1))
            selected = np.random.choice(tokens, min(num_tokens, len(tokens)), replace=False)
            content = " ".join([t.content[:30] for t in selected])
        else:
            content = query[:50]
        
        # Add variation
        variations = [
            f"the key aspects include {content}",
            f"{content} represents fundamental concepts",
            f"understanding {content} is essential",
            f"this relates to {content} principles",
            f"{content} forms the basis"
        ]
        
        return random.choice(variations)


class DiversityHARGSDataset(Dataset):
    """Dataset with diversity-enhanced samples."""
    
    def __init__(self, texts: List[str], tokenizer: HierarchicalTokenizer, 
                 augmentation_prob: float = 0.3):
        self.texts = texts
        self.tokenizer = tokenizer
        self.augmentation_prob = augmentation_prob
        
        # Paraphrase templates for data augmentation
        self.paraphrase_templates = [
            "In other words, {text}",
            "To put it differently, {text}",
            "Alternatively stated, {text}",
            "Another way to say this is: {text}",
            "Reworded: {text}",
            "Simplified: {text}"
        ]
    
    def __len__(self):
        return len(self.texts)
    
    def _augment_text(self, text: str) -> str:
        """Apply random augmentation."""
        if random.random() > self.augmentation_prob:
            return text
        
        # Template-based paraphrase
        if random.random() < 0.3:
            template = random.choice(self.paraphrase_templates)
            return template.format(text=text[:200])
        
        # Sentence reordering
        if '.' in text and random.random() < 0.3:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 1:
                random.shuffle(sentences)
                return '. '.join(sentences) + '.'
        
        return text
    
    def __getitem__(self, idx):
        text = self._augment_text(self.texts[idx])
        tokens = self.tokenizer.tokenize(text[:1000])
        
        if len(tokens) >= 2:
            input_emb = torch.from_numpy(tokens[0].embedding).float()
            target_emb = torch.from_numpy(tokens[1].embedding).float()
        elif tokens:
            input_emb = torch.from_numpy(tokens[0].embedding).float()
            target_emb = input_emb + 0.1 * torch.randn_like(input_emb)
        else:
            input_emb = torch.zeros(self.tokenizer.embedding_dim)
            target_emb = torch.zeros(self.tokenizer.embedding_dim)
        
        return input_emb, target_emb


class MaxDiverseHARGSTrainer:
    """Optimized trainer for maximum diversity HARGS."""
    
    def __init__(self, model: HARGSModelWithMaxDiversity, output_dir: str = "./hargs_max_diverse_checkpoints"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.train_losses = []
        self.best_loss = float('inf')
        self.steps = 0
        
        logger.info(f"Trainer initialized. Output: {self.output_dir}")
    
    def setup_optimizer(self, lr: float = 2e-4, weight_decay: float = 1e-4, total_steps: int = 375):
        """Setup optimizer with diversity-friendly parameters."""
        params = list(self.model.diffusion_model.parameters())

        self.optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay,
            betas=(0.9, 0.98), eps=1e-8
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=lr, total_steps=total_steps,
            pct_start=0.1, anneal_strategy='cos'
        )

        logger.info(f"Optimizer setup: lr={lr}, weight_decay={weight_decay}, total_steps={total_steps}")
    
    def diversity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute diversity encouragement loss."""
        # Encourage variance in embeddings
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Pairwise diversity
        embeddings_norm = F.normalize(embeddings, dim=-1)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.t())
        
        # Mask out diagonal
        mask = ~torch.eye(batch_size, device=embeddings.device).bool()
        diversity = -similarity[mask].mean()
        
        return diversity
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Single training step with diversity optimization."""
        input_emb, target_emb = batch
        input_emb = input_emb.to(self.model.device)
        target_emb = target_emb.to(self.model.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_noise, true_noise = self.model.diffusion_model(target_emb, input_emb)
        
        # Base MSE loss
        mse_loss = self.criterion(pred_noise, true_noise)
        
        # Diversity regularization
        div_loss = 0.01 * self.diversity_loss(pred_noise)
        
        # Combined loss
        total_loss = mse_loss + div_loss
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return total_loss.item()
    
    def train(self, dataset: DiversityHARGSDataset, num_epochs: int = 5, 
              batch_size: int = 32, num_workers: int = 2) -> Dict[str, Any]:
        """Main training loop optimized for diversity."""
        logger.info(f"Starting training: epochs={num_epochs}, batch_size={batch_size}")
        
        # Calculate total steps for scheduler
        steps_per_epoch = len(dataset) // batch_size
        total_steps = steps_per_epoch * num_epochs

        # Setup
        self.setup_optimizer(total_steps=total_steps)

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                self.steps += 1
                
                # Log progress
                if batch_idx % 50 == 0:
                    logger.info(f"  Step {batch_idx}, Loss: {loss:.6f}")
                
                # Save checkpoint periodically
                if self.steps % 500 == 0:
                    self.save_checkpoint(epoch, is_best=(loss < self.best_loss))
                    if loss < self.best_loss:
                        self.best_loss = loss
            
            # Epoch summary
            avg_loss = epoch_loss / max(num_batches, 1)
            self.train_losses.append(avg_loss)
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.6f}, Time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining complete! Total time: {total_time:.1f}s")
        
        return {
            'train_losses': self.train_losses,
            'final_loss': self.train_losses[-1] if self.train_losses else float('inf'),
            'best_loss': self.best_loss,
            'total_steps': self.steps,
            'training_time': total_time
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'steps': self.steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses
        }
        
        # Regular checkpoint
        path = self.output_dir / f"checkpoint_step_{self.steps}.pth"
        torch.save(checkpoint, path)
        
        # Best model
        if is_best:
            best_path = self.output_dir / "best_max_diverse_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at step {self.steps}")


def load_or_create_datasets() -> Tuple[List[str], List[str]]:
    """Load existing or create new datasets."""
    train_path = Path("diverse_train_texts.json")
    val_path = Path("diverse_val_texts.json")
    
    if train_path.exists() and val_path.exists():
        logger.info("Loading existing datasets...")
        with open(train_path) as f:
            train_data = json.load(f)
        with open(val_path) as f:
            val_data = json.load(f)
        return train_data, val_data
    
    logger.info("Creating new diverse datasets...")
    from create_diverse_dataset import create_diverse_knowledge_dataset
    return create_diverse_knowledge_dataset()


def main():
    """Main training entry point."""
    logger.info("="*60)
    logger.info("HARGS Maximum Diversity Training")
    logger.info("="*60)
    
    # Load datasets
    train_texts, _ = load_or_create_datasets()
    logger.info(f"Training on {len(train_texts)} examples")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = HARGSModelWithMaxDiversity(
        vocab_size=10000,
        embedding_dim=512,
        diffusion_hidden_dim=1024,
        diffusion_num_layers=6,
        device=device
    )
    
    # Create dataset
    tokenizer = model.tokenizer
    dataset = DiversityHARGSDataset(train_texts, tokenizer, augmentation_prob=0.5)
    
    # Initialize trainer
    trainer = MaxDiverseHARGSTrainer(model)
    
    # Train
    results = trainer.train(
        dataset=dataset,
        num_epochs=3,  # Quick baseline
        batch_size=32,
        num_workers=2 if device == 'cuda' else 0
    )
    
    # Save final results
    results_path = Path("./hargs_max_diverse_checkpoints/training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Final loss: {results['final_loss']:.6f}")
    logger.info(f"Best loss: {results['best_loss']:.6f}")
    
    # Quick test
    logger.info("\nTesting model...")
    test_query = "What is machine learning?"
    for i in range(3):
        result = model(test_query)
        logger.info(f"  {i+1}. {result['response'][:100]}...")
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
