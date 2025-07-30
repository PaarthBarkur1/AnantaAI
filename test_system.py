#!/usr/bin/env python3
"""
Test script for AnantaAI QA System
"""

import sys
import time
import logging
from qna import ContextAgent, QAAgent, SentenceTransformer, QAConfig, JSONContentSource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic QA functionality"""
    print("🧪 Testing AnantaAI QA System...")
    
    try:
        # Initialize components
        print("📋 Initializing QA system...")
        config = QAConfig()
        context_agent = ContextAgent(config)
        
        # Add data source
        print("📊 Loading data sources...")
        context_agent.add_source(JSONContentSource("context.json"))
        
        # Build index
        print("🔍 Building semantic index...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        context_agent.build_semantic_search_index(embedder)
        
        # Initialize QA agent
        print("🤖 Initializing QA agent...")
        qa_agent = QAAgent(device=-1, config=config)  # CPU mode for testing
        
        # Test queries
        test_queries = [
            "What are the eligibility criteria?",
            "Tell me about placement statistics",
            "How long is the program?",
            "What is the fee structure?"
        ]
        
        print("\n🔍 Testing queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            start_time = time.time()
            
            result = qa_agent.process_query(query, context_agent, top_k=3)
            
            processing_time = time.time() - start_time
            
            print(f"   ✅ Answer: {result['answer'][:100]}...")
            print(f"   ⏱️  Time: {processing_time:.2f}s")
            print(f"   📊 Confidence: {result.get('confidence', 0):.2f}")
            print(f"   📚 Sources: {len(result.get('metadata', {}).get('sources_used', []))}")
        
        print("\n✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        return False

def test_data_sources():
    """Test data source loading"""
    print("\n🗂️  Testing data sources...")
    
    try:
        import json
        import os
        
        # Check required files
        required_files = ["context.json", "sources.json", "faq_data.py"]
        for file in required_files:
            if os.path.exists(file):
                print(f"   ✅ {file} exists")
            else:
                print(f"   ❌ {file} missing")
                return False
        
        # Validate JSON files
        with open("context.json", 'r', encoding='utf-8') as f:
            context_data = json.load(f)
            print(f"   📊 Context entries: {len(context_data)}")
        
        with open("sources.json", 'r', encoding='utf-8') as f:
            sources_data = json.load(f)
            print(f"   🌐 Web sources: {len(sources_data)}")
        
        print("   ✅ Data sources validation passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Data sources test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AnantaAI System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Data Sources", test_data_sources),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
