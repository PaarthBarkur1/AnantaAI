#!/usr/bin/env python3
"""
Test script to verify the embedding model upgrades work correctly
"""

import sys
import os
import time
from typing import List

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.qna import ContextAgent, QAConfig, JSONContentSource
from backend.embedding_config import EmbeddingModelManager
from sentence_transformers import SentenceTransformer


def test_model_configuration():
    """Test the model configuration system"""
    print("🧪 Testing Model Configuration System")
    print("-" * 50)
    
    # Test QAConfig
    config = QAConfig()
    print(f"✅ Default model: {config.embedding_model}")
    print(f"✅ Model validation: {config.validate_embedding_model()}")
    
    # Test recommendations
    prod_model = config.get_recommended_model("production")
    dev_model = config.get_recommended_model("development")
    print(f"✅ Production recommendation: {prod_model}")
    print(f"✅ Development recommendation: {dev_model}")
    
    # Test model info
    model_info = EmbeddingModelManager.get_model_config(config.embedding_model)
    if model_info:
        print(f"✅ Model info: {model_info.description} ({model_info.dimensions}D)")
    
    return True


def test_embedding_loading():
    """Test loading different embedding models"""
    print("\n🧪 Testing Embedding Model Loading")
    print("-" * 50)
    
    # Test models (start with smaller ones for speed)
    test_models = [
        "sentence-transformers/all-MiniLM-L12-v2",  # Fast to load
        "BAAI/bge-large-en-v1.5"  # Best performance
    ]
    
    for model_name in test_models:
        try:
            print(f"Loading {model_name}...")
            start_time = time.time()
            
            # Load model
            embedder = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            
            # Test encoding
            test_text = "What are the eligibility criteria for admission?"
            embedding = embedder.encode([test_text])
            
            print(f"✅ {model_name}")
            print(f"   Load time: {load_time:.2f}s")
            print(f"   Embedding shape: {embedding.shape}")
            
        except Exception as e:
            print(f"❌ {model_name}: {e}")
            return False
    
    return True


def test_search_functionality():
    """Test the enhanced search functionality"""
    print("\n🧪 Testing Enhanced Search Functionality")
    print("-" * 50)
    
    try:
        # Initialize system
        config = QAConfig(embedding_model="sentence-transformers/all-MiniLM-L12-v2")  # Use faster model for testing
        context_agent = ContextAgent(config)
        
        # Load test data
        data_file = "backend/context.json"
        if not os.path.exists(data_file):
            print(f"❌ Test data file not found: {data_file}")
            return False
        
        context_agent.add_source(JSONContentSource(data_file))
        faq_data = context_agent.get_faq_data()
        print(f"✅ Loaded {len(faq_data)} documents")
        
        # Build index
        print("Building search index...")
        embedder = SentenceTransformer(config.embedding_model)
        context_agent.build_semantic_search_index(embedder)
        print("✅ Index built successfully")
        
        # Test search
        test_queries = [
            "What are the eligibility criteria?",
            "Tell me about placements",
            "How long is the program?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Test regular search
            start_time = time.time()
            results = context_agent.search(query, top_k=3, use_hybrid=False)
            search_time = time.time() - start_time
            
            print(f"  Regular search: {len(results)} results in {search_time*1000:.1f}ms")
            if results:
                print(f"  Top result confidence: {results[0].confidence:.3f}")
            
            # Test hybrid search (if available)
            if hasattr(context_agent, 'cosine_index'):
                start_time = time.time()
                hybrid_results = context_agent.search(query, top_k=3, use_hybrid=True)
                hybrid_time = time.time() - start_time
                
                print(f"  Hybrid search: {len(hybrid_results)} results in {hybrid_time*1000:.1f}ms")
                if hybrid_results:
                    print(f"  Top result confidence: {hybrid_results[0].confidence:.3f}")
        
        print("✅ Search functionality working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """Test performance comparison between old and new models"""
    print("\n🧪 Testing Performance Comparison")
    print("-" * 50)
    
    # Compare old vs new model
    models_to_compare = [
        ("all-MiniLM-L6-v2", "Legacy model"),
        ("sentence-transformers/all-MiniLM-L12-v2", "Improved model"),
    ]
    
    test_query = "What are the eligibility criteria for admission?"
    
    for model_name, description in models_to_compare:
        try:
            print(f"\nTesting {description}: {model_name}")
            
            # Initialize
            config = QAConfig(embedding_model=model_name)
            context_agent = ContextAgent(config)
            context_agent.add_source(JSONContentSource("backend/context.json"))
            
            # Build index and measure time
            start_time = time.time()
            embedder = SentenceTransformer(model_name)
            context_agent.build_semantic_search_index(embedder)
            index_time = time.time() - start_time
            
            # Test search
            start_time = time.time()
            results = context_agent.search(test_query, top_k=3)
            search_time = time.time() - start_time
            
            # Results
            avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
            
            print(f"  Index build time: {index_time:.2f}s")
            print(f"  Search time: {search_time*1000:.1f}ms")
            print(f"  Results found: {len(results)}")
            print(f"  Average confidence: {avg_confidence:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return True


def main():
    """Run all tests"""
    print("🚀 AnantaAI Embedding Upgrade Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Configuration", test_model_configuration),
        ("Embedding Loading", test_embedding_loading),
        ("Search Functionality", test_search_functionality),
        ("Performance Comparison", test_performance_comparison),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"🏆 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Embedding upgrade is working correctly.")
        print("\n💡 Next steps:")
        print("1. Run 'python choose_embedding_model.py' to optimize for your system")
        print("2. Run 'python benchmark_embeddings.py' to see performance gains")
        print("3. Check EMBEDDING_UPGRADE_GUIDE.md for detailed information")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Tests cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
