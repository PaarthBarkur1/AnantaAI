#!/usr/bin/env python3
"""
Test script to verify the improved QA logic and answer quality enhancements.
This script tests the enhanced prompt engineering, context processing, answer validation,
and semantic search improvements.
"""

import sys
import os
import time
from typing import List, Dict

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from backend.qna import ContextAgent, QAAgent, QAConfig, JSONContentSource
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running this from the project root directory")
    sys.exit(1)


def test_enhanced_qa_system():
    """Test the enhanced QA system with various query types"""
    
    print("🧪 Testing Enhanced QA System Logic")
    print("=" * 50)
    
    # Initialize enhanced configuration
    config = QAConfig(
        enable_query_expansion=True,
        enable_context_reranking=True,
        context_fusion_strategy="intelligent",
        answer_quality_threshold=0.15,
        max_search_results=8
    )
    
    # Initialize components
    print("📋 Initializing enhanced QA system...")
    context_agent = ContextAgent(config)
    
    # Add data sources
    print("📊 Loading data sources...")
    try:
        context_agent.add_source(JSONContentSource("backend/context.json"))
        print(f"✅ Loaded {len(context_agent.get_faq_data())} FAQ entries")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Build semantic index with optimized model
    print("🔍 Building semantic index...")
    try:
        # Use a lightweight model for testing
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        context_agent.build_semantic_search_index(embedder)
        print("✅ Semantic index built successfully")
    except Exception as e:
        print(f"❌ Failed to build index: {e}")
        return False
    
    # Initialize QA agent
    print("🤖 Initializing enhanced QA agent...")
    try:
        qa_agent = QAAgent(device=-1, config=config)  # CPU mode for testing
        print("✅ QA agent initialized")
    except Exception as e:
        print(f"❌ Failed to initialize QA agent: {e}")
        return False
    
    # Test queries with different types and complexities
    test_queries = [
        # Eligibility queries
        {
            "query": "What are the eligibility criteria for M.Mgt?",
            "type": "eligibility",
            "expected_keywords": ["degree", "bachelor", "engineering", "60%", "marks"]
        },
        {
            "query": "What is the minimum CGPA required?",
            "type": "eligibility", 
            "expected_keywords": ["cgpa", "minimum", "60%"]
        },
        
        # Placement queries
        {
            "query": "What is the average placement package?",
            "type": "placement",
            "expected_keywords": ["ctc", "salary", "average", "lpa", "₹"]
        },
        {
            "query": "Which companies recruit from IISc M.Mgt?",
            "type": "placement",
            "expected_keywords": ["companies", "wells fargo", "jpmc", "uber"]
        },
        
        # Admission queries
        {
            "query": "What is the CAT cutoff for general category?",
            "type": "admission",
            "expected_keywords": ["cat", "percentile", "98.8", "cutoff"]
        },
        
        # Curriculum queries
        {
            "query": "How many credits are required for the program?",
            "type": "curriculum",
            "expected_keywords": ["credits", "64", "total"]
        },
        
        # Campus life queries
        {
            "query": "What is student life like at IISc?",
            "type": "campus",
            "expected_keywords": ["campus", "classes", "facilities", "accommodation"]
        },
        
        # Complex/specific queries
        {
            "query": "Tell me about the interview process and what to expect",
            "type": "interview",
            "expected_keywords": ["interview", "group discussion", "math", "probability"]
        }
    ]
    
    print("\n🎯 Testing Enhanced Query Processing")
    print("-" * 40)
    
    results = []
    total_time = 0
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        query_type = test_case["type"]
        expected_keywords = test_case["expected_keywords"]
        
        print(f"\n{i}. Testing {query_type} query:")
        print(f"   Query: {query}")
        
        # Process query
        start_time = time.time()
        try:
            response = qa_agent.process_query(query, context_agent, top_k=5)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            answer = response.get("answer", "")
            confidence = response.get("confidence", 0.0)
            metadata = response.get("metadata", {})
            
            print(f"   Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Processing time: {processing_time:.3f}s")
            print(f"   Method: {metadata.get('method', 'unknown')}")
            
            # Evaluate answer quality
            quality_score = evaluate_answer_quality(answer, expected_keywords, query_type)
            print(f"   Quality score: {quality_score:.2f}/5.0")
            
            results.append({
                "query": query,
                "type": query_type,
                "answer": answer,
                "confidence": confidence,
                "quality_score": quality_score,
                "processing_time": processing_time,
                "method": metadata.get('method', 'unknown')
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "query": query,
                "type": query_type,
                "error": str(e),
                "quality_score": 0.0,
                "processing_time": 0.0
            })
    
    # Generate summary report
    print("\n📊 Enhanced QA System Performance Report")
    print("=" * 50)
    
    successful_queries = [r for r in results if "error" not in r]
    failed_queries = [r for r in results if "error" in r]
    
    if successful_queries:
        avg_confidence = sum(r["confidence"] for r in successful_queries) / len(successful_queries)
        avg_quality = sum(r["quality_score"] for r in successful_queries) / len(successful_queries)
        avg_time = sum(r["processing_time"] for r in successful_queries) / len(successful_queries)
        
        print(f"✅ Successful queries: {len(successful_queries)}/{len(test_queries)}")
        print(f"📈 Average confidence: {avg_confidence:.3f}")
        print(f"🎯 Average quality score: {avg_quality:.2f}/5.0")
        print(f"⚡ Average processing time: {avg_time:.3f}s")
        print(f"🕒 Total processing time: {total_time:.3f}s")
        
        # Method distribution
        methods = {}
        for r in successful_queries:
            method = r.get("method", "unknown")
            methods[method] = methods.get(method, 0) + 1
        
        print(f"\n🔧 Answer generation methods:")
        for method, count in methods.items():
            print(f"   {method}: {count} queries")
        
        # Quality by query type
        print(f"\n📋 Quality by query type:")
        type_quality = {}
        for r in successful_queries:
            qtype = r["type"]
            if qtype not in type_quality:
                type_quality[qtype] = []
            type_quality[qtype].append(r["quality_score"])
        
        for qtype, scores in type_quality.items():
            avg_score = sum(scores) / len(scores)
            print(f"   {qtype}: {avg_score:.2f}/5.0 (n={len(scores)})")
    
    if failed_queries:
        print(f"\n❌ Failed queries: {len(failed_queries)}")
        for r in failed_queries:
            print(f"   - {r['query']}: {r['error']}")
    
    # Overall assessment
    success_rate = len(successful_queries) / len(test_queries)
    if success_rate >= 0.9 and avg_quality >= 3.5:
        print(f"\n🎉 EXCELLENT: Enhanced QA system performing very well!")
    elif success_rate >= 0.8 and avg_quality >= 3.0:
        print(f"\n✅ GOOD: Enhanced QA system performing well with room for improvement")
    elif success_rate >= 0.6:
        print(f"\n⚠️ FAIR: Enhanced QA system needs optimization")
    else:
        print(f"\n❌ POOR: Enhanced QA system requires significant improvements")
    
    return success_rate >= 0.8


def evaluate_answer_quality(answer: str, expected_keywords: List[str], query_type: str) -> float:
    """Evaluate the quality of an answer based on various criteria"""
    if not answer or len(answer.strip()) < 10:
        return 0.0
    
    score = 0.0
    answer_lower = answer.lower()
    
    # Keyword relevance (0-2 points)
    keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
    keyword_score = min(2.0, (keyword_matches / len(expected_keywords)) * 2.0)
    score += keyword_score
    
    # Answer completeness (0-1 point)
    if len(answer.split()) >= 10:
        score += 0.5
    if len(answer.split()) >= 20:
        score += 0.5
    
    # Specificity (0-1 point)
    import re
    if re.search(r'\d+|₹|%|specific|exactly|approximately', answer):
        score += 1.0
    
    # Coherence (0-1 point)
    if answer.endswith(('.', '!', '?')) and not any(phrase in answer_lower for phrase in 
                                                   ["i don't know", "i cannot", "sorry"]):
        score += 1.0
    
    return min(5.0, score)


if __name__ == "__main__":
    try:
        success = test_enhanced_qa_system()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
