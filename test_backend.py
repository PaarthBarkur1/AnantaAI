#!/usr/bin/env python3
"""
Test script to verify the backend API is working correctly.
"""

import requests
import json
import time

def test_backend_api():
    """Test the backend API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing AnantaAI Backend API")
    print("=" * 40)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("[OK] Backend server is running")
        else:
            print(f"[FAIL] Server responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Cannot connect to backend server: {e}")
        print("Make sure the backend is running with: python backend/main.py")
        return False
    
    # Test 2: Test query endpoint
    try:
        query_data = {
            "text": "What are the eligibility criteria for M.Mgt?",
            "max_results": 3
        }
        
        print(f"[TEST] Sending query: {query_data['text']}")
        response = requests.post(
            f"{base_url}/api/query",
            json=query_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("[OK] Query processed successfully")
            print(f"     Answer length: {len(result.get('answer', ''))}")
            print(f"     Confidence: {result.get('confidence', 0)}")
            print(f"     Sources: {len(result.get('sources', []))}")
            print(f"     Processing time: {result.get('processing_time', 0):.2f}s")
            return True
        else:
            print(f"[FAIL] Query failed with status {response.status_code}")
            print(f"       Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Query request failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"[FAIL] Invalid JSON response: {e}")
        return False

def main():
    """Main test function."""
    success = test_backend_api()
    
    print("\n" + "=" * 40)
    if success:
        print("SUCCESS: Backend API is working correctly!")
        return 0
    else:
        print("FAILED: Backend API has issues")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
