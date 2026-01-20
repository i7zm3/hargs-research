"""
Final evaluation script for the HARGS diverse response model.
Tests the trained model's ability to generate varied responses and meet the 80% diversity target.
"""

import torch
import json
from pathlib import Path
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple
import logging

# Import the model components
from enhanced_hargs_diverse_train import HARGSModelWithMaxDiversity, MaxDiverseHARGSTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_path: str) -> HARGSModelWithMaxDiversity:
    """Load the trained diverse model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}...")
    
    # Initialize model with same parameters as training
    model = HARGSModelWithMaxDiversity(
        vocab_size=10000,
        embedding_dim=512,
        diffusion_hidden_dim=1024,
        diffusion_num_layers=6
    )
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Model loaded successfully!")
    return model

def test_response_diversity(model: HARGSModelWithMaxDiversity, query: str, num_tests: int = 100) -> Dict:
    """Test the diversity of responses for a single query."""
    logger.info(f"Testing diversity for query: '{query}' with {num_tests} repetitions...")
    
    responses = []
    latencies = []
    confidences = []
    
    for i in range(num_tests):
        result = model(query)
        responses.append(result['response'])
        latencies.append(result['latency'])
        confidences.append(result['confidence'])
        
        if (i + 1) % 20 == 0:
            logger.info(f"Completed {i + 1}/{num_tests} tests...")
    
    # Calculate diversity metrics
    unique_responses = len(set(responses))
    diversity_ratio = unique_responses / num_tests
    response_counts = Counter(responses)
    most_common_response = response_counts.most_common(1)[0] if response_counts else ("", 0)
    
    # Calculate semantic diversity using simple hash-based approach
    response_hashes = [hash(r) for r in responses]
    hash_variance = np.var(response_hashes) if response_hashes else 0
    
    metrics = {
        'query': query,
        'total_responses': num_tests,
        'unique_responses': unique_responses,
        'diversity_ratio': diversity_ratio,
        'avg_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'avg_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'most_common_response': most_common_response,
        'hash_variance': hash_variance,
        'responses': responses,
        'latencies': latencies,
        'confidences': confidences
    }
    
    return metrics

def evaluate_model_comprehensive(model: HARGSModelWithMaxDiversity) -> Dict:
    """Comprehensive evaluation of the model's diversity performance."""
    test_queries = [
        "What is machine learning?",
        "Explain quantum computing briefly",
        "How does photosynthesis work?",
        "Solve x^2 + 2*x - 3 = 0",
        "What is artificial intelligence?",
        "Describe blockchain technology",
        "How does DNA replication work?",
        "Explain neural networks"
    ]
    
    all_metrics = []
    overall_unique = 0
    overall_total = 0
    
    for query in test_queries:
        metrics = test_response_diversity(model, query, num_tests=50)  # Using 50 for faster testing
        all_metrics.append(metrics)
        
        overall_unique += metrics['unique_responses']
        overall_total += metrics['total_responses']
        
        logger.info(f"Query: '{query}'")
        logger.info(f"  Diversity: {metrics['diversity_ratio']:.3f} ({metrics['unique_responses']}/{metrics['total_responses']})")
        logger.info(f"  Avg Latency: {metrics['avg_latency']:.3f}s")
        logger.info(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")
        logger.info(f"  Most Common Length: {len(metrics['most_common_response'][0])}")
        logger.info("")  # Fixed: Added empty string
    
    overall_diversity = overall_unique / overall_total if overall_total > 0 else 0
    
    comprehensive_metrics = {
        'individual_query_metrics': all_metrics,
        'overall_diversity_ratio': overall_diversity,
        'total_unique_responses': overall_unique,
        'total_tested_responses': overall_total,
        'num_tested_queries': len(test_queries),
        'target_diversity_met': overall_diversity >= 0.80,
        'diversity_percentage': overall_diversity * 100
    }
    
    return comprehensive_metrics

def main():
    """Main evaluation function."""
    logger.info("Starting comprehensive evaluation of HARGS diverse model...")
    
    # Load the trained model
    checkpoint_path = "./hargs_max_diverse_checkpoints/best_max_diverse_model.pth"
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    model = load_trained_model(checkpoint_path)
    
    # Evaluate the model comprehensively
    results = evaluate_model_comprehensive(model)
    
    # Print final results
    logger.info("="*80)
    logger.info("COMPREHENSIVE EVALUATION RESULTS")
    logger.info("="*80)
    
    logger.info(f"Overall Diversity: {results['diversity_percentage']:.1f}% ({results['total_unique_responses']}/{results['total_tested_responses']})")
    logger.info(f"Target (80%) Met: {'YES' if results['target_diversity_met'] else 'NO'}")
    logger.info(f"Number of tested queries: {results['num_tested_queries']}")
    
    # Check SLA metrics if available
    try:
        with open("./hargs_max_diverse_checkpoints/max_diverse_training_results.json", "r") as f:
            training_results = json.load(f)
            sla_met = training_results.get('sla_met', False)
            diversity_met = training_results.get('diversity_met', False)
            logger.info(f"Training reported SLA met: {sla_met}")
            logger.info(f"Training reported diversity target met: {diversity_met}")
    except FileNotFoundError:
        logger.info("Training results file not found")
    
    # Test a few sample interactions to demonstrate variety
    logger.info("\nSAMPLE INTERACTIONS (demonstrating variety):")
    logger.info("-" * 50)
    
    sample_query = "What is machine learning?"
    sample_results = test_response_diversity(model, sample_query, num_tests=10)
    
    print(f"\nQuery: {sample_query}")
    print(f"Diversity: {sample_results['diversity_ratio']:.3f} ({sample_results['unique_responses']}/10)")
    print("\nSample responses:")
    for i, response in enumerate(sample_results['responses'][:10], 1):
        print(f"  {i:2d}. {response[:100]}...")
    
    # Save detailed results
    results_path = Path("./hargs_max_diverse_checkpoints/final_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {results_path}")
    logger.info(f"Final diversity achieved: {results['diversity_percentage']:.1f}%")
    
    # Success criteria
    if results['diversity_percentage'] >= 80:
        logger.info("\n🎉 SUCCESS: Model achieved 80%+ diversity target!")
    elif results['diversity_percentage'] >= 70:
        logger.info("\n👍 GOOD: Model achieved 70-79% diversity - close to target!")
    elif results['diversity_percentage'] >= 50:
        logger.info("\n✅ OK: Model achieved 50-69% diversity - moderate improvement!")
    else:
        logger.info("\n⚠️ NEEDS IMPROVEMENT: Model achieved <50% diversity - further tuning required!")

if __name__ == "__main__":
    main()
