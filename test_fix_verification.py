#!/usr/bin/env python3
"""
Quick test to verify the ContextAgent attribute error fix
"""

import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from backend.qna import ContextAgent, QAAgent, QAConfig, JSONContentSource
    from sentence_transformers import SentenceTransformer
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality to ensure no attribute errors"""
    
    print("🧪 Testing basic functionality...")
    
    try:
        # Initialize configuration
        config = QAConfig()
        print("✅ QAConfig created")
        
        # Initialize context agent
        context_agent = ContextAgent(config)
        print("✅ ContextAgent created")
        
        # Add data source
        try:
            context_agent.add_source(JSONContentSource("backend/context.json"))
            print("✅ Data source added")
        except Exception as e:
            print(f"⚠️ Data source failed: {e}")
            return False
        
        # Build semantic index
        try:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            context_agent.build_semantic_search_index(embedder)
            print("✅ Semantic index built")
        except Exception as e:
            print(f"❌ Index building failed: {e}")
            return False
        
        # Initialize QA agent
        try:
            qa_agent = QAAgent(device=-1, config=config)  # CPU mode
            print("✅ QAAgent created")
        except Exception as e:
            print(f"❌ QAAgent creation failed: {e}")
            return False
        
        # Test a simple query
        try:
            test_query = "What are the eligibility criteria?"
            print(f"🔍 Testing query: {test_query}")
            
            response = qa_agent.process_query(test_query, context_agent, top_k=3)
            
            if response and "answer" in response:
                answer = response["answer"]
                confidence = response.get("confidence", 0.0)
                print(f"✅ Query processed successfully")
                print(f"   Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                print(f"   Confidence: {confidence:.3f}")
                return True
            else:
                print("❌ No valid response received")
                return False
                
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_agent_methods():
    """Test specific ContextAgent methods that were modified"""
    
    print("\n🔧 Testing ContextAgent methods...")
    
    try:
        config = QAConfig()
        context_agent = ContextAgent(config)
        
        # Test query preprocessing
        test_queries = [
            "What are eligibility criteria?",
            "Tell me about placements",
            "How much are the fees?"
        ]
        
        for query in test_queries:
            try:
                processed = context_agent._preprocess_query(query)
                print(f"✅ Preprocessed '{query}' -> '{processed[:50]}...'")
            except Exception as e:
                print(f"❌ Preprocessing failed for '{query}': {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ ContextAgent methods test failed: {e}")
        return False

def test_qa_agent_methods():
    """Test specific QAAgent methods that were modified"""
    
    print("\n🤖 Testing QAAgent methods...")
    
    try:
        config = QAConfig()
        qa_agent = QAAgent(device=-1, config=config)
        
        # Test query intent analysis
        test_queries = [
            "What are the eligibility criteria?",
            "How do I apply?",
            "Tell me about placement statistics"
        ]
        
        for query in test_queries:
            try:
                intent = qa_agent._analyze_query_intent(query.lower())
                print(f"✅ Analyzed '{query}' -> type: {intent['type']}, specificity: {intent['specificity']}")
            except Exception as e:
                print(f"❌ Intent analysis failed for '{query}': {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ QAAgent methods test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Verifying ContextAgent Attribute Error Fix")
    print("=" * 50)
    
    success = True
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test specific methods
    if not test_context_agent_methods():
        success = False
    
    if not test_qa_agent_methods():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The attribute error should be fixed.")
    else:
        print("❌ Some tests failed. There may still be issues.")
    
    sys.exit(0 if success else 1)
