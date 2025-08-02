#!/usr/bin/env python3
"""
Test script for improved QA system
Tests various types of questions that students and recruiters would ask about IISc M.Mgt
"""

import sys
import os
import time
import json
from typing import List, Dict, Any

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.qna import ContextAgent, QAAgent, QAConfig, JSONContentSource, WebContentSource
from sentence_transformers import SentenceTransformer
import torch

def test_questions() -> List[Dict[str, str]]:
    """Define test questions covering various aspects of IISc M.Mgt"""
    return [
        # Admission and Eligibility Questions
        {
            "category": "Admissions",
            "question": "What are the eligibility criteria for IISc M.Mgt?",
            "expected_keywords": ["bachelor", "engineering", "60%", "marks", "cgpa", "stem"]
        },
        {
            "category": "Admissions", 
            "question": "What is the CAT cutoff for IISc M.Mgt?",
            "expected_keywords": ["98.8", "percentile", "cutoff", "general"]
        },
        {
            "category": "Admissions",
            "question": "Is work experience required for admission?",
            "expected_keywords": ["not mandatory", "no formal advantage", "help during interviews"]
        },
        
        # Academic Questions
        {
            "category": "Academics",
            "question": "What is the course structure of M.Mgt program?",
            "expected_keywords": ["64 credits", "hard core", "stream core", "electives", "project"]
        },
        {
            "category": "Academics",
            "question": "What courses are offered in the curriculum?",
            "expected_keywords": ["probability", "statistics", "operations", "marketing", "economics"]
        },
        {
            "category": "Academics",
            "question": "What is the class schedule like?",
            "expected_keywords": ["4 days", "monday to thursday", "9:30 AM to 2 PM", "fridays"]
        },
        
        # Placement Questions
        {
            "category": "Placements",
            "question": "What is the average placement package?",
            "expected_keywords": ["₹27 LPA", "average", "ctc", "placement"]
        },
        {
            "category": "Placements",
            "question": "Which companies recruit from IISc M.Mgt?",
            "expected_keywords": ["wells fargo", "jpmc", "uber", "quantum street", "companies"]
        },
        {
            "category": "Placements",
            "question": "What types of roles do graduates get?",
            "expected_keywords": ["data science", "quantitative finance", "ai research", "analytics"]
        },
        
        # Financial Questions
        {
            "category": "Finances",
            "question": "What is the tuition fee for the program?",
            "expected_keywords": ["₹2,50,000", "per year", "tuition", "fee"]
        },
        {
            "category": "Finances",
            "question": "Are scholarships available?",
            "expected_keywords": ["no institute scholarships", "external scholarships", "assistantships"]
        },
        
        # Campus Life Questions
        {
            "category": "Campus Life",
            "question": "What are the accommodation facilities?",
            "expected_keywords": ["single-room", "hostel", "campus", "library", "canteens"]
        },
        {
            "category": "Campus Life",
            "question": "What is the overall student experience like?",
            "expected_keywords": ["chill", "balanced", "ample free time", "vibrant campus"]
        },
        
        # Specific Technical Questions
        {
            "category": "Admissions",
            "question": "What is the interview process like?",
            "expected_keywords": ["statistics", "math", "probability", "group discussion", "coding"]
        },
        {
            "category": "Academics",
            "question": "What electives are available?",
            "expected_keywords": ["macroeconomics", "deep learning", "entrepreneurship", "simulation"]
        },
        {
            "category": "Placements",
            "question": "Do candidates from specific backgrounds get higher packages?",
            "expected_keywords": ["cse", "it", "sde", "above ₹30 LPA", "engineering"]
        }
    ]

def initialize_qa_system():
    """Initialize the QA system with all data sources"""
    print("🚀 Initializing QA system...")
    
    # Create configuration
    config = QAConfig(
        use_ai_generation=False,  # Disable AI generation for testing
        min_confidence=0.1,
        max_search_results=8
    )
    
    # Initialize context agent
    context_agent = ContextAgent(config)
    
    # Add JSON FAQ source
    context_agent.add_source(JSONContentSource("backend/context.json"))
    
    # Add structured FAQ data
    from backend.faq_data import FAQ_DATA
    faq_count = 0
    for category, questions in FAQ_DATA.items():
        for question, answer in questions.items():
            faq_entry = {
                "question": question,
                "answer": answer.strip(),
                "metadata": {
                    "source": "json",
                    "category": category,
                    "type": "faq"
                }
            }
            context_agent.faq_data.append(faq_entry)
            faq_count += 1
    
    print(f"📚 Loaded {faq_count} FAQ entries from {len(FAQ_DATA)} categories")
    
    # Initialize embedding model
    print("🔍 Loading embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    context_agent.build_semantic_search_index(embedder)
    
    # Initialize QA agent
    print("🤖 Initializing QA agent...")
    qa_agent = QAAgent(device=-1, config=config)  # Use CPU for testing
    
    return context_agent, qa_agent

def evaluate_answer_quality(answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """Evaluate the quality of an answer"""
    answer_lower = answer.lower()
    
    # Check for expected keywords
    found_keywords = []
    missing_keywords = []
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    # Calculate keyword coverage
    keyword_coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
    
    # Check answer length
    word_count = len(answer.split())
    length_score = 1.0 if 10 <= word_count <= 100 else 0.5 if 5 <= word_count <= 150 else 0.2
    
    # Check for domain relevance
    domain_terms = ['iisc', 'm.mgt', 'management', 'bangalore', 'placement', 'ctc', 'curriculum']
    domain_relevance = sum(1 for term in domain_terms if term in answer_lower) / len(domain_terms)
    
    # Check for specific information
    has_numbers = any(char.isdigit() for char in answer)
    has_specific_terms = any(term in answer_lower for term in ['specific', 'include', 'consist', 'require'])
    
    # Overall quality score
    quality_score = (
        keyword_coverage * 0.4 +
        length_score * 0.2 +
        domain_relevance * 0.2 +
        (0.1 if has_numbers else 0.0) +
        (0.1 if has_specific_terms else 0.0)
    )
    
    return {
        "keyword_coverage": keyword_coverage,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "word_count": word_count,
        "length_score": length_score,
        "domain_relevance": domain_relevance,
        "has_numbers": has_numbers,
        "has_specific_terms": has_specific_terms,
        "quality_score": quality_score
    }

def run_tests():
    """Run comprehensive tests on the QA system"""
    print("🧪 Starting QA System Tests")
    print("=" * 60)
    
    # Initialize system
    context_agent, qa_agent = initialize_qa_system()
    
    # Get test questions
    test_cases = test_questions()
    
    # Results storage
    results = []
    total_quality_score = 0
    
    print(f"\n📝 Testing {len(test_cases)} questions...")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        category = test_case["category"]
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        
        print(f"\n{i:2d}. [{category}] {question}")
        print("-" * 50)
        
        # Process query
        start_time = time.time()
        result = qa_agent.process_query(question, context_agent, top_k=5)
        processing_time = time.time() - start_time
        
        # Extract answer
        answer = result.get("answer", "")
        confidence = result.get("confidence", 0.0)
        
        # Evaluate quality
        quality_metrics = evaluate_answer_quality(answer, expected_keywords)
        
        # Store results
        test_result = {
            "question": question,
            "category": category,
            "answer": answer,
            "confidence": confidence,
            "processing_time": processing_time,
            "quality_metrics": quality_metrics
        }
        results.append(test_result)
        
        # Print results
        print(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Processing Time: {processing_time:.2f}s")
        print(f"Quality Score: {quality_metrics['quality_score']:.1%}")
        print(f"Keyword Coverage: {quality_metrics['keyword_coverage']:.1%}")
        print(f"Found Keywords: {', '.join(quality_metrics['found_keywords'][:3])}")
        if quality_metrics['missing_keywords']:
            print(f"Missing Keywords: {', '.join(quality_metrics['missing_keywords'][:3])}")
        
        total_quality_score += quality_metrics['quality_score']
    
    # Calculate overall statistics
    avg_quality_score = total_quality_score / len(test_cases)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    avg_processing_time = sum(r["processing_time"] for r in results) / len(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Questions Tested: {len(test_cases)}")
    print(f"Average Quality Score: {avg_quality_score:.1%}")
    print(f"Average Confidence: {avg_confidence:.1%}")
    print(f"Average Processing Time: {avg_processing_time:.2f}s")
    
    # Category-wise analysis
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result["quality_metrics"]["quality_score"])
    
    print(f"\n📈 Category-wise Performance:")
    for category, scores in categories.items():
        avg_score = sum(scores) / len(scores)
        print(f"  {category}: {avg_score:.1%} ({len(scores)} questions)")
    
    # Save detailed results
    with open("test_results.json", "w") as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json.dump({
            "summary": {
                "total_questions": len(test_cases),
                "average_quality_score": float(avg_quality_score),
                "average_confidence": float(avg_confidence),
                "average_processing_time": float(avg_processing_time)
            },
            "category_performance": {
                cat: float(sum(scores) / len(scores)) for cat, scores in categories.items()
            },
            "detailed_results": convert_numpy_types(results)
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to test_results.json")
    
    # Quality assessment
    if avg_quality_score >= 0.8:
        print("✅ EXCELLENT: System provides high-quality, accurate answers")
    elif avg_quality_score >= 0.6:
        print("✅ GOOD: System provides satisfactory answers with room for improvement")
    elif avg_quality_score >= 0.4:
        print("⚠️  FAIR: System needs improvement in answer quality")
    else:
        print("❌ POOR: System requires significant improvements")
    
    return results

if __name__ == "__main__":
    try:
        results = run_tests()
        print("\n🎉 Testing completed successfully!")
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
