#!/usr/bin/env python3
"""
Test script to verify the chat interface improvements
"""

import streamlit as st
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_chat_display():
    """Test the chat display functionality"""
    
    st.title("🧪 Chat Interface Test")
    
    # Simulate a user question and AI response
    test_data = {
        'text': 'The eligibility criteria for the IISc Master of Management program include having a bachelor\'s degree in engineering, science, or technology with a minimum of 60% marks. Candidates must also have a valid GATE score or qualify through the institute\'s entrance examination.',
        'query': 'What are the eligibility criteria for admission?',
        'processing_time': 1.23,
        'confidence': 0.87,
        'sources_count': 3,
        'sources': [
            {'source': 'Official Website', 'confidence': 0.9, 'url': 'https://mgmt.iisc.ac.in'},
            {'source': 'Admission Brochure', 'confidence': 0.85},
            {'source': 'FAQ Document', 'confidence': 0.8}
        ]
    }
    
    st.markdown("## Test Chat Display")
    st.markdown("This shows how the user question and AI response should appear:")
    
    # Simulate the chat interface
    st.markdown("---")
    
    # Show user's question
    if test_data.get("query"):
        st.markdown("### 💬 Your Question")
        st.markdown(
            f'<div class="user-question">{test_data["query"]}</div>', 
            unsafe_allow_html=True
        )
        st.markdown("")  # Add some spacing
    
    st.markdown("### 🤖 AI Response")
    st.markdown(
        f'<div class="answer-box">{test_data["text"]}</div>', 
        unsafe_allow_html=True
    )
    
    # Metrics
    if test_data.get('confidence') is not None:
        confidence_display = f"{test_data['confidence']:.1%}"
        metrics_html = f"""
            <div class="metric-row">
                <span>⏱ {test_data["processing_time"]:.2f}s</span>
                <span>📊 {confidence_display} confidence</span>
                <span>📚 {test_data["sources_count"]} sources</span>
            </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Sources
    if test_data["sources"]:
        with st.expander("View Sources"):
            for i, source in enumerate(test_data["sources"], 1):
                st.write(f"**{i}.** {source.get('source', 'Unknown')}")
                if source.get("url"):
                    st.write(f"🔗 [View]({source['url']})")
                if source.get("confidence"):
                    st.write(f"Confidence: {source['confidence']:.1%}")
                st.divider()

# Add the CSS styles
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
    
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #6b7280;
    }
    
    .metric-row span {
        background: #f3f4f6;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    test_chat_display()
