#!/usr/bin/env python3
"""
Embedding Model Benchmarking Tool for AnantaAI

This script benchmarks different embedding models on your specific dataset
to help you choose the best one for your use case.
"""

import sys
import os
import time
import json
from typing import List, Dict, Any
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.qna import ContextAgent, QAConfig, JSONContentSource
from backend.embedding_config import EmbeddingModelManager
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingBenchmark:
    """Benchmark different embedding models"""
    
    def __init__(self, data_file: str = "backend/context.json"):
        self.data_file = data_file
        self.test_queries = [
            "What are the eligibility criteria?",
            "Tell me about placement statistics",
            "How long is the program?",
            "What is the fee structure?",
            "What are the core subjects?",
            "How is the admission process?",
            "What companies visit for placements?",
            "What is the duration of the course?"
        ]
        self.results = {}
    
    def load_test_data(self) -> bool:
        """Load test data"""
        try:
            config = QAConfig()
            self.context_agent = ContextAgent(config)
            self.context_agent.add_source(JSONContentSource(self.data_file))
            return len(self.context_agent.get_faq_data()) > 0
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return False
    
    def benchmark_model(self, model_name: str) -> Dict[str, Any]:
        """Benchmark a specific model"""
        print(f"\n🔍 Benchmarking: {model_name}")
        
        try:
            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            start_time = time.time()
            embedder = SentenceTransformer(model_name, device=device)
            load_time = time.time() - start_time
            
            # Build index
            start_time = time.time()
            self.context_agent.build_semantic_search_index(embedder)
            index_time = time.time() - start_time
            
            # Test search performance
            search_times = []
            relevance_scores = []
            
            for query in self.test_queries:
                start_time = time.time()
                results = self.context_agent.search(query, top_k=3)
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                # Calculate relevance score (simplified)
                if results:
                    avg_confidence = np.mean([r.confidence for r in results])
                    relevance_scores.append(avg_confidence)
                else:
                    relevance_scores.append(0.0)
            
            # Calculate metrics
            avg_search_time = np.mean(search_times)
            avg_relevance = np.mean(relevance_scores)
            
            # Memory usage (approximate)
            model_config = EmbeddingModelManager.get_model_config(model_name)
            estimated_memory = model_config.dimensions * len(self.context_agent.get_faq_data()) * 4 / (1024**2)  # MB
            
            result = {
                "model_name": model_name,
                "load_time": load_time,
                "index_time": index_time,
                "avg_search_time": avg_search_time,
                "avg_relevance": avg_relevance,
                "estimated_memory_mb": estimated_memory,
                "dimensions": model_config.dimensions if model_config else "unknown",
                "status": "success"
            }
            
            print(f"   ✅ Load time: {load_time:.2f}s")
            print(f"   ✅ Index time: {index_time:.2f}s") 
            print(f"   ✅ Avg search time: {avg_search_time*1000:.1f}ms")
            print(f"   ✅ Avg relevance: {avg_relevance:.3f}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {
                "model_name": model_name,
                "status": "failed",
                "error": str(e)
            }
    
    def run_benchmark(self, models: List[str] = None) -> Dict[str, Any]:
        """Run benchmark on specified models"""
        if models is None:
            # Default set of models to benchmark
            models = [
                "BAAI/bge-large-en-v1.5",
                "sentence-transformers/all-mpnet-base-v2", 
                "sentence-transformers/all-MiniLM-L12-v2",
                "all-MiniLM-L6-v2"
            ]
        
        print("🚀 Starting Embedding Model Benchmark")
        print("=" * 50)
        
        if not self.load_test_data():
            print("❌ Failed to load test data")
            return {}
        
        print(f"📊 Loaded {len(self.context_agent.get_faq_data())} documents")
        print(f"🔍 Testing with {len(self.test_queries)} queries")
        
        results = {}
        for model_name in models:
            results[model_name] = self.benchmark_model(model_name)
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report"""
        report = "\n" + "=" * 60 + "\n"
        report += "🏆 EMBEDDING MODEL BENCHMARK REPORT\n"
        report += "=" * 60 + "\n"
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if v.get("status") == "success"}
        
        if not successful_results:
            report += "❌ No successful benchmarks to report\n"
            return report
        
        # Summary table
        report += "\n📊 PERFORMANCE SUMMARY\n"
        report += "-" * 60 + "\n"
        report += f"{'Model':<35} {'Load(s)':<8} {'Search(ms)':<10} {'Relevance':<10}\n"
        report += "-" * 60 + "\n"
        
        for model_name, result in successful_results.items():
            short_name = model_name.split('/')[-1][:30]
            report += f"{short_name:<35} {result['load_time']:<8.1f} {result['avg_search_time']*1000:<10.1f} {result['avg_relevance']:<10.3f}\n"
        
        # Detailed analysis
        report += "\n🔍 DETAILED ANALYSIS\n"
        report += "-" * 60 + "\n"
        
        # Best performers
        best_relevance = max(successful_results.items(), key=lambda x: x[1]['avg_relevance'])
        fastest_search = min(successful_results.items(), key=lambda x: x[1]['avg_search_time'])
        fastest_load = min(successful_results.items(), key=lambda x: x[1]['load_time'])
        
        report += f"🎯 Best Relevance: {best_relevance[0]} ({best_relevance[1]['avg_relevance']:.3f})\n"
        report += f"⚡ Fastest Search: {fastest_search[0]} ({fastest_search[1]['avg_search_time']*1000:.1f}ms)\n"
        report += f"🚀 Fastest Load: {fastest_load[0]} ({fastest_load[1]['load_time']:.1f}s)\n"
        
        # Recommendations
        report += "\n💡 RECOMMENDATIONS\n"
        report += "-" * 60 + "\n"
        
        if best_relevance[1]['avg_relevance'] > 0.7:
            report += f"🏆 For Production: {best_relevance[0]} (highest accuracy)\n"
        
        if fastest_search[1]['avg_search_time'] < 0.01:
            report += f"⚡ For Speed: {fastest_search[0]} (fastest search)\n"
        
        # Find balanced option
        balanced_scores = {}
        for model_name, result in successful_results.items():
            # Normalize scores (0-1) and combine
            max_relevance = max(r['avg_relevance'] for r in successful_results.values())
            min_search_time = min(r['avg_search_time'] for r in successful_results.values())
            max_search_time = max(r['avg_search_time'] for r in successful_results.values())
            
            relevance_score = result['avg_relevance'] / max_relevance
            speed_score = 1 - ((result['avg_search_time'] - min_search_time) / (max_search_time - min_search_time))
            
            balanced_scores[model_name] = (relevance_score + speed_score) / 2
        
        best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
        report += f"⚖️  For Balance: {best_balanced[0]} (best overall)\n"
        
        # Failed models
        failed_results = {k: v for k, v in results.items() if v.get("status") == "failed"}
        if failed_results:
            report += "\n❌ FAILED MODELS\n"
            report += "-" * 60 + "\n"
            for model_name, result in failed_results.items():
                report += f"{model_name}: {result.get('error', 'Unknown error')}\n"
        
        report += "\n" + "=" * 60 + "\n"
        return report


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark embedding models for AnantaAI")
    parser.add_argument("--models", nargs="+", help="Specific models to benchmark")
    parser.add_argument("--data", default="backend/context.json", help="Path to test data file")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with fewer models")
    
    args = parser.parse_args()
    
    # Select models to benchmark
    if args.models:
        models = args.models
    elif args.quick:
        models = [
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]
    else:
        models = [
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-large-v2", 
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "all-MiniLM-L6-v2"
        ]
    
    # Run benchmark
    benchmark = EmbeddingBenchmark(args.data)
    results = benchmark.run_benchmark(models)
    
    # Generate and display report
    report = benchmark.generate_report(results)
    print(report)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Benchmark cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
