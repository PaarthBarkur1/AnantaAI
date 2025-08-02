#!/usr/bin/env python3
"""
Simple test to verify chat interface functionality
"""

import streamlit as st

# Initialize session state
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None

st.title("🧪 Simple Chat Test")

# Add CSS
st.markdown("""
<style>
    .user-question {
        background: #374151;
        color: #e5e7eb;
        border-left: 4px solid #10b981;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 1rem;
        font-weight: 500;
        line-height: 1.5;
    }
    
    .answer-box {
        background: #1f2937;
        color: #f9fafb;
        border-left: 4px solid #60a5fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        white-space: pre-wrap;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Simple input
user_input = st.text_input("Ask a question:", key="test_input")

if st.button("Send") and user_input:
    # Simulate processing
    st.session_state.current_answer = {
        'text': f"This is a test response to your question: '{user_input}'. The chat interface should show both your question and this response.",
        'query': user_input,
        'processing_time': 0.5,
        'confidence': 0.95,
        'sources_count': 2,
        'sources': []
    }
    st.rerun()

# Display chat
if st.session_state.current_answer:
    answer_data = st.session_state.current_answer
    
    st.markdown("---")
    
    # Debug info
    st.write("**Debug Info:**")
    st.write(f"Query stored: '{answer_data.get('query', 'NOT FOUND')}'")
    st.write(f"Answer keys: {list(answer_data.keys())}")
    
    # Show user's question
    user_query = answer_data.get("query", "").strip()
    if user_query:
        st.markdown("### 💬 Your Question")
        st.markdown(
            f'<div class="user-question">{user_query}</div>', 
            unsafe_allow_html=True
        )
    else:
        st.error("❌ User question not found!")
    
    # Show AI response
    st.markdown("### 🤖 AI Response")
    st.markdown(
        f'<div class="answer-box">{answer_data["text"]}</div>', 
        unsafe_allow_html=True
    )

# Test buttons
st.markdown("---")
st.markdown("### Test Buttons")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Test Question 1"):
        st.session_state.current_answer = {
            'text': "This is a test response to question 1.",
            'query': "What is test question 1?",
            'processing_time': 0.3,
            'confidence': 0.9,
            'sources_count': 1,
            'sources': []
        }
        st.rerun()

with col2:
    if st.button("Test Question 2"):
        st.session_state.current_answer = {
            'text': "This is a test response to question 2.",
            'query': "What is test question 2?",
            'processing_time': 0.4,
            'confidence': 0.85,
            'sources_count': 2,
            'sources': []
        }
        st.rerun()

with col3:
    if st.button("Clear"):
        st.session_state.current_answer = None
        st.rerun()

st.markdown("---")
st.markdown("**Instructions:**")
st.markdown("1. Type a question in the input box and click 'Send'")
st.markdown("2. You should see both your question and the AI response")
st.markdown("3. Try the test buttons to see if they work")
st.markdown("4. Check the debug info to see what's being stored")
