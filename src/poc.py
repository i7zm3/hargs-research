"""
HARGS Proof of Concept Implementation

This module implements a minimal working prototype of the HARGS architecture,
focusing on the key innovations described in the whitepaper:
1. Hierarchical tokenization
2. Split-half negation diffusion
3. Usage-weighted discrimination
4. Monarch coordinator
5. Symbolic-Neural integration
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math
from abc import ABC, abstractmethod


class TokenType(Enum):
    WORD = "word"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    DOCUMENT = "document"


@dataclass
class Token:
    """Represents a semantic token in the HARGS system."""
    id: str
    content: str
    level: TokenType
    embedding: np.ndarray
    usage_weight: float = 0.0
    quality_score: float = 0.5
    last_access: float = 0.0
    access_count: int = 0
    cluster_id: int = 0


class Tokenizer:
    """Implements hierarchical tokenization with semantic embeddings."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.embedding_dim = 512  # Will adapt based on usage weight
        
    def tokenize(self, text: str) -> List[Token]:
        """Tokenizes text at multiple levels based on the HARGS hierarchy."""
        tokens = []
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        for para_idx, para in enumerate(paragraphs):
            if para.strip():
                # Create paragraph token
                para_embedding = self.generate_embedding(para, TokenType.PARAGRAPH)
                para_token = Token(
                    id=f"para_{para_idx}",
                    content=para,
                    level=TokenType.PARAGRAPH,
                    embedding=para_embedding
                )
                tokens.append(para_token)
                
                # Split paragraph into sentences
                sentences = [s.strip() for s in para.split('.') if s.strip()]
                for sent_idx, sent in enumerate(sentences):
                    # Create sentence token
                    sent_embedding = self.generate_embedding(sent, TokenType.SENTENCE)
                    sent_token = Token(
                        id=f"sentence_{para_idx}_{sent_idx}",
                        content=sent + '.',
                        level=TokenType.SENTENCE,
                        embedding=sent_embedding
                    )
                    tokens.append(sent_token)
                    
                    # Split sentence into words
                    words = sent.split()
                    for word_idx, word in enumerate(words):
                        # Create word token
                        word_embedding = self.generate_embedding(word, TokenType.WORD)
                        word_token = Token(
                            id=f"word_{para_idx}_{sent_idx}_{word_idx}",
                            content=word,
                            level=TokenType.WORD,
                            embedding=word_embedding
                        )
                        tokens.append(word_token)
        
        return tokens
    
    def generate_embedding(self, content: str, level: TokenType) -> np.ndarray:
        """Generate semantic embedding for content."""
        # This would normally use a trained encoder
        # For PoC, we'll simulate based on content and level
        # Different dimensions based on level (as described in the paper)
        
        # Determine embedding dimension based on a mock usage weight
        # Higher usage gets higher dimension (Section 3.1)
        mock_usage_weight = hash(content) % 100 / 100.0  # Mock usage simulation
        
        if mock_usage_weight > 0.8:
            emb_dim = 1024  # Hot tokens
        elif mock_usage_weight > 0.4:
            emb_dim = 512   # Warm tokens
        else:
            emb_dim = 256   # Cold tokens
            
        # Generate embedding based on content and level
        hash_val = hash(content + level.value)
        np.random.seed(abs(hash_val) % (2**32 - 1))
        embedding = np.random.randn(emb_dim)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm
        
        return embedding


class SplitHalfDiffusion:
    """Implements the split-half negation diffusion process."""
    
    def __init__(self, embedding_dim: int = 512):
        self.embed_dim = embedding_dim
        self.pos_dim = embedding_dim // 2
        self.neg_dim = embedding_dim // 2
        
        # Initialize split transformation matrices (learning parameters)
        self.W_pos = np.random.randn(self.pos_dim, self.embed_dim)
        self.W_neg = np.random.randn(self.neg_dim, self.embed_dim)
        self.b_pos = np.random.randn(self.pos_dim)
        self.b_neg = np.random.randn(self.neg_dim)
        
        # Initialize recombination parameters
        self.W_combine = np.random.randn(self.embed_dim, self.pos_dim + self.neg_dim)
        
        # Diffusion parameters
        self.T = 50  # Number of diffusion steps
        self.beta_min = 0.0001
        self.beta_max = 0.02
    
    def ensure_dimensions(self, embedding: np.ndarray) -> np.ndarray:
        """Ensure the embedding has the expected dimension."""
        if len(embedding) != self.embed_dim:
            # Adjust dimensions to match expected size
            if len(embedding) < self.embed_dim:
                # Pad with zeros
                padded = np.zeros(self.embed_dim)
                padded[:len(embedding)] = embedding
                return padded
            else:
                # Truncate to expected size
                return embedding[:self.embed_dim]
        return embedding
        
    def split_embedding(self, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split embedding into positive and negative semantic directions."""
        e_adjusted = self.ensure_dimensions(e)
        e_pos = self.W_pos @ e_adjusted + self.b_pos
        e_neg = self.W_neg @ e_adjusted + self.b_neg
        return e_pos, e_neg
    
    def recombine_embeddings(self, e_pos: np.ndarray, e_neg: np.ndarray) -> np.ndarray:
        """Recombine positive and negative embeddings."""
        combined_input = np.concatenate([e_pos, e_neg])
        if len(combined_input) != self.pos_dim + self.neg_dim:
            # Adjust dimensions if needed
            expected_len = self.pos_dim + self.neg_dim
            if len(combined_input) < expected_len:
                padded = np.zeros(expected_len)
                padded[:len(combined_input)] = combined_input
                combined_input = padded
            else:
                combined_input = combined_input[:expected_len]
        recombined = self.W_combine @ combined_input
        return recombined
    
    def get_noise_schedule(self) -> List[float]:
        """Generate noise schedule as defined in the paper."""
        betas = []
        for t in range(1, self.T + 1):
            beta_t = self.beta_min + (self.beta_max - self.beta_min) * (t / self.T) ** 2
            betas.append(beta_t)
        return betas
    
    def forward_process(self, e_0: np.ndarray) -> List[np.ndarray]:
        """Forward diffusion process (noising)."""
        betas = self.get_noise_schedule()
        e_t = e_0.copy()
        trajectory = [e_0]
        
        for beta_t in betas:
            # q(e_t | e_{t-1}) = N(e_t; √(1-β_t) e_{t-1}, β_t I)
            e_t = np.sqrt(1 - beta_t) * e_t + np.sqrt(beta_t) * np.random.randn(*e_t.shape)
            trajectory.append(e_t)
        
        return trajectory
    
    def denoise_step(self, e_t: np.ndarray, t: int, t_emb: Optional[np.ndarray] = None) -> np.ndarray:
        """Single denoising step (simplified - in practice would use neural network)."""
        betas = self.get_noise_schedule()
        if t >= len(betas):
            return e_t
            
        beta_t = betas[t]
        alpha_t = 1.0 - beta_t
        alpha_bar_t = self._alpha_bar(t + 1)  # cumulative noise
        alpha_bar_t_prev = self._alpha_bar(t)  # previous cumulative noise
        
        # Ensure e_t has correct dimensions before processing
        e_t = self.ensure_dimensions(e_t)
        
        # Simplified denoising (would normally use learned neural network)
        # This is a mock implementation showing the concept
        e_t_prev = (e_t - np.sqrt(beta_t) * np.random.randn(*e_t.shape)) / np.sqrt(alpha_t)
        
        # Add conditional guidance based on split-half concept
        e_pos, e_neg = self.split_embedding(e_t)
        
        # Apply asymmetric weighting based on timestep (from Section 4.3)
        w_pos = 1 - t / self.T  # decreasing
        w_neg = t / self.T      # increasing
        
        # Reconstruct with weighted components
        guided_e_t = w_pos * (self.W_pos.T @ e_pos) + w_neg * (self.W_neg.T @ e_neg)
        
        # Combine with denoised version
        result = 0.7 * e_t_prev + 0.3 * guided_e_t
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm != 0:
            result = result / norm
            
        return result
    
    def _alpha_bar(self, t: int) -> float:
        """Calculate cumulative noise."""
        betas = self.get_noise_schedule()
        if t == 0:
            return 1.0
        product = 1.0
        for i in range(min(t, len(betas))):
            product *= (1 - betas[i])
        return product
    
    def sample(self, e_0: np.ndarray, condition: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a sample using the diffusion process."""
        # Forward process - noising
        trajectory = self.forward_process(e_0)
        e_noisy = trajectory[-1]
        
        # Reverse process - denoising
        e_t = e_noisy.copy()
        for t in range(self.T - 1, -1, -1):
            e_t = self.denoise_step(e_t, t)
        
        return e_t


class UsageWeightCalculator:
    """Calculates usage-weighted discrimination as described in Section 5."""
    
    def __init__(self):
        self.access_counts = {}
        self.last_access_times = {}
        self.quality_scores = {}
        self.max_count = 1  # Will be updated dynamically
        
    def update_access(self, token_id: str, time: float, feedback: Optional[float] = None):
        """Update access statistics for a token."""
        if token_id not in self.access_counts:
            self.access_counts[token_id] = 0
            self.quality_scores[token_id] = 0.5
        
        self.access_counts[token_id] += 1
        self.last_access_times[token_id] = time
        
        # Update quality score if feedback provided (0-1 scale)
        if feedback is not None:
            # Exponential moving average update
            self.quality_scores[token_id] = 0.9 * self.quality_scores[token_id] + 0.1 * feedback
        
        # Update max count for normalization
        self.max_count = max(self.max_count, self.access_counts[token_id])
    
    def calculate_weight(self, token_id: str, time: float) -> float:
        """Calculate the complete usage weight for a token."""
        # Base frequency (Section 5.1)
        count = self.access_counts.get(token_id, 0)
        f_base = math.log(1 + count) / math.log(1 + self.max_count) if self.max_count > 0 else 0.0
        
        # Temporal decay (Section 5.1)
        last_access = self.last_access_times.get(token_id, time)
        delta_t = time - last_access
        
        # Multiple timescales
        alpha_hourly, alpha_daily = 0.5, 0.3
        alpha_weekly, alpha_monthly = 0.15, 0.05
        lambda_hourly, lambda_daily = 0.05, 0.02
        lambda_weekly, lambda_monthly = 0.01, 0.001
        
        # Convert time difference to different scales
        delta_hours = delta_t
        delta_days = delta_hours / 24
        delta_weeks = delta_days / 7
        delta_months = delta_days / 30  # Approx
        
        f_decay = (
            alpha_hourly * math.exp(-lambda_hourly * delta_hours) +
            alpha_daily * math.exp(-lambda_daily * delta_days) +
            alpha_weekly * math.exp(-lambda_weekly * delta_weeks) +
            alpha_monthly * math.exp(-lambda_monthly * delta_months)
        )
        
        # Quality score (Section 5.1)
        f_quality = self.quality_scores.get(token_id, 0.5)
        
        # Combined weight
        weight = f_base * f_decay * f_quality
        
        # Normalize to [0, 1) to prevent unbounded growth (Section 5.2)
        normalized_weight = weight / (1 + weight)
        
        return normalized_weight


class MonarchCoordinator:
    """Lightweight meta-controller for orchestrating subsystems."""
    
    def __init__(self):
        self.gpu_allocation = 1.0  # 100% of available GPU initially
        self.memory_budget = 1.0   # 100% of available memory initially
        self.time_limits = {'fast': 0.01, 'reasoning': 0.04, 'deep': 0.20}  # in seconds
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine appropriate strategy."""
        # Simple complexity heuristics (Section 7.3)
        length = len(query)
        complexity = min(length / 100, 1.0)  # Normalize to [0, 1]
        
        # Heuristic: longer and more complex queries need reasoning/deep processing
        if complexity < 0.3:
            strategy = 'fast'
        elif complexity < 0.7:
            strategy = 'reasoning'
        else:
            strategy = 'deep'
        
        return {
            'complexity': complexity,
            'strategy': strategy,
            'estimated_resources': self.calculate_resource_needs(strategy)
        }
    
    def calculate_resource_needs(self, strategy: str) -> Dict[str, float]:
        """Calculate resource needs based on strategy."""
        if strategy == 'fast':
            return {
                'gpu_fraction': 0.1,      # 10% GPU
                'time_limit': self.time_limits['fast'],
                'memory_fraction': 0.1
            }
        elif strategy == 'reasoning':
            return {
                'gpu_fraction': 0.4,      # 40% GPU
                'time_limit': self.time_limits['reasoning'],
                'memory_fraction': 0.4
            }
        else:  # deep
            return {
                'gpu_fraction': 0.8,      # 80% GPU
                'time_limit': self.time_limits['deep'],
                'memory_fraction': 0.8
            }
    
    def allocate_resources(self, strategy: str, available_resources: Dict[str, float]) -> Dict[str, float]:
        """Allocate resources based on strategy."""
        needs = self.calculate_resource_needs(strategy)
        
        allocated = {}
        for resource, need in needs.items():
            resource_name = resource.replace('_fraction', '')
            if resource_name in available_resources:
                allocated[resource_name] = need * available_resources[resource_name]
        
        return allocated


class SymbolicEngine:
    """Placeholder for symbolic reasoning engine (Section 9)."""
    
    def __init__(self):
        pass
    
    def solve_mathematical_expression(self, expr: str) -> Any:
        """Simple mathematical solver."""
        # This would interface with SymPy in a real implementation
        # For PoC, we'll handle a simple case
        if "x^2" in expr and "=" in expr:
            # Handle x^2 + ax + b = 0
            try:
                # Parse simple quadratic: e.g. "x^2 + 3*x - 4 = 0"
                import re
                coeffs = [float(x) for x in re.findall(r'-?\d+\.?\d*', expr)[:3]]
                if len(coeffs) >= 3:
                    a, b, c = 1, coeffs[0], coeffs[1]  # Assuming coefficient of x^2 is 1
                    discriminant = b**2 - 4*a*c
                    if discriminant >= 0:
                        x1 = (-b + math.sqrt(discriminant)) / (2*a)
                        x2 = (-b - math.sqrt(discriminant)) / (2*a)
                        return {'roots': [x1, x2], 'valid': True}
            except:
                pass
        return {'roots': None, 'valid': False}
    
    def simple_logic_solver(self, premises: List[str], query: str) -> bool:
        """Simplified logical reasoning."""
        # This would interface with an automated theorem prover
        # For PoC: simple propositional logic
        all_premises = " ".join(premises).lower()
        query_lower = query.lower()
        
        # Simple pattern matching for "all X are Y" and "Z is X" -> "Z is Y"
        import re
        
        # Patterns: "all humans are mortal", "socrates is human" -> "socrates is mortal"
        for premise in premises:
            if "all" in premise.lower() and "are" in premise.lower():
                pattern = r"all (\w+) are (\w+)"
                match = re.search(pattern, premise.lower())
                if match:
                    category1, category2 = match.groups()
                    # Check if query follows the pattern "Z is category1" -> infer "Z is category2"
                    query_pattern = rf"(\w+) is {category1}"
                    query_match = re.search(query_pattern, query.lower())
                    if query_match:
                        subject = query_match.group(1)
                        # Since "subject is category1" and "all category1 are category2",
                        # we can conclude "subject is category2"
                        return True
        
        return False


class HARGS_PoC:
    """Proof of concept implementation of the HARGS architecture."""
    
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.diffusion_model = SplitHalfDiffusion()
        self.weight_calculator = UsageWeightCalculator()
        self.monarch = MonarchCoordinator()
        self.symbolic_engine = SymbolicEngine()
        self.tokens = {}  # Token storage
        
        # Available resources
        self.available_resources = {
            'gpu': 100.0,  # percentage
            'memory': 100.0,  # percentage
            'time': float('inf')  # seconds
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using the HARGS architecture."""
        # Step 1: Monarch analyzes query and determines strategy
        analysis = self.monarch.analyze_query(query)
        strategy = analysis['strategy']
        
        print(f"Query: '{query}'")
        print(f"Strategy selected: {strategy}")
        print(f"Complexity: {analysis['complexity']:.2f}")
        
        # Step 2: Allocate resources based on strategy
        resources = self.monarch.allocate_resources(strategy, self.available_resources)
        print(f"Resources allocated: {resources}")
        
        # Step 3: Tokenization
        tokens = self.tokenizer.tokenize(query)
        print(f"Generated {len(tokens)} tokens")
        
        # Step 4: Select relevant tokens based on weights and similarity
        if tokens:
            primary_token = tokens[0]  # Take the first token as primary
            primary_embedding = primary_token.embedding
            
            # Update access counts
            self.weight_calculator.update_access(primary_token.id, time=0, feedback=None)
            
            # Calculate weights for other tokens
            weighted_tokens = []
            for token in tokens[:10]:  # Consider first 10 tokens for efficiency
                weight = self.weight_calculator.calculate_weight(token.id, time=0)
                weighted_tokens.append((token, weight))
            
            # Sort by weight (highest first)
            weighted_tokens.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Top token weight: {weighted_tokens[0][1]:.3f}" if weighted_tokens else "No tokens")
        
        # Step 5: Processing based on strategy
        result = None
        if strategy == 'fast':
            # Fast path: RAG + Diffusion (Section 1.1)
            if tokens:
                # Use diffusion to generate response
                sampled_embedding = self.diffusion_model.sample(primary_embedding)
                result = {
                    'response': f"Fast response based on diffused embedding of shape {sampled_embedding.shape}",
                    'confidence': 0.8,
                    'strategy_used': 'fast'
                }
        elif strategy == 'reasoning':
            # Reasoning path: Symbolic + Chain + Diffusion (Section 1.1)
            # For PoC, we'll try symbolic reasoning first
            symbolic_result = self.symbolic_engine.solve_mathematical_expression(query)
            if symbolic_result['valid']:
                result = {
                    'response': f"Symbolic solution found: {symbolic_result['roots']}",
                    'confidence': 0.9,
                    'strategy_used': 'reasoning'
                }
            else:
                # Fall back to diffusion if symbolic doesn't apply
                sampled_embedding = self.diffusion_model.sample(primary_embedding)
                result = {
                    'response': f"Diffusion-generated response for complex query",
                    'confidence': 0.7,
                    'strategy_used': 'reasoning'
                }
        else:  # deep
            # Deep path: Multiple strategies in parallel (Section 10.4)
            # For PoC, we'll run both symbolic and diffusion approaches
            symbolic_result = self.symbolic_engine.solve_mathematical_expression(query)
            sampled_embedding = self.diffusion_model.sample(primary_embedding)
            
            if symbolic_result['valid']:
                result = {
                    'response': f"Deep analysis: Symbolic solution {symbolic_result['roots']} and diffused embedding",
                    'confidence': 0.95,
                    'strategy_used': 'deep'
                }
            else:
                result = {
                    'response': f"Deep analysis: Diffusion-generated response",
                    'confidence': 0.85,
                    'strategy_used': 'deep'
                }
        
        print(f"Response: {result['response']}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 50)
        
        return result


def main():
    """Run the HARGS proof of concept."""
    print("=" * 60)
    print("HARGS PROOF OF CONCEPT IMPLEMENTATION")
    print("=" * 60)
    print()
    
    # Initialize the system
    hargs = HARGS_PoC()
    
    # Test queries demonstrating different aspects of HARGS
    test_queries = [
        "What is 2 + 2?",  # Simple arithmetic (fast)
        "Solve x^2 + 3*x - 4 = 0",  # Quadratic equation (reasoning)
        "How does photosynthesis work in plants? Give a detailed explanation.",  # Complex query (deep)
        "Translate 'hello world' to French.",  # Translation (fast)
        "If all humans are mortal and Socrates is human, is Socrates mortal?"  # Logic puzzle (reasoning)
    ]
    
    results = []
    for i, query in enumerate(test_queries):
        print(f"Test {i+1}:")
        result = hargs.process_query(query)
        results.append(result)
        print()
    
    print("=" * 60)
    print("SUMMARY OF RESULTS:")
    print("=" * 60)
    
    for i, result in enumerate(results):
        print(f"Test {i+1}: Strategy={result['strategy_used']}, Confidence={result['confidence']:.2f}")
    
    print()
    print("The proof of concept demonstrates the key HARGS concepts:")
    print("- Hierarchical tokenization")
    print("- Split-half negation diffusion") 
    print("- Usage-weighted discrimination")
    print("- Monarch-coordinated strategy selection")
    print("- Symbolic-neural integration")


if __name__ == "__main__":
    main()