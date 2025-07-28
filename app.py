import streamlit as st
import torch
from qna import ContextAgent, JSONContentSource, WebContentSource, QAAgent, SentenceTransformer

# Page config
st.set_page_config(
    page_title="IISc M.Mgt QA System",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'context_agent' not in st.session_state:
    st.session_state.context_agent = None
if 'qa_agent' not in st.session_state:
    st.session_state.qa_agent = None

# Title and description
st.title("IISc M.Mgt QA System 🎓")
st.markdown("""
This system can answer your questions about the IISc M.Mgt program using:
- FAQ data from JSON files
- Web-scraped content from URLs
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # FAQ file input
    faq_file = st.text_input("FAQ JSON file path (optional)", 
                            value="context.json",
                            help="Path to your FAQ JSON file")
    
    # URL input
    url = st.text_input("URL to scrape (optional)", 
                       help="Enter a URL to scrape content from")
    
    # Model settings
    st.subheader("Model Settings")
    device = st.radio("Device", 
                     options=["GPU", "CPU"],
                     index=0 if torch.cuda.is_available() else 1,
                     help="Select the device to run the model on")
    
    top_k = st.slider("Number of contexts to retrieve", 
                      min_value=1, 
                      max_value=10, 
                      value=5,
                      help="Number of relevant contexts to use for answering")
    
    # Initialize/Reset button
    if st.button("Initialize/Reset System"):
        with st.spinner("Initializing the system..."):
            # Initialize context agent
            context_agent = ContextAgent()
            
            # Add sources
            if faq_file:
                try:
                    context_agent.add_source(JSONContentSource(faq_file))
                except Exception as e:
                    st.error(f"Error loading FAQ file: {str(e)}")
            
            if url:
                try:
                    context_agent.add_source(WebContentSource(url))
                except Exception as e:
                    st.error(f"Error scraping URL: {str(e)}")
            
            if context_agent.get_faq_data():
                # Initialize embedder and build index
                embedder = SentenceTransformer('all-MiniLM-L6-v2')
                context_agent.build_semantic_search_index(embedder)
                st.session_state.context_agent = context_agent
                
                # Initialize QA agent
                device_id = 0 if device == "GPU" else -1
                st.session_state.qa_agent = QAAgent(device=device_id)
                
                st.success("System initialized successfully!")
            else:
                st.error("No content sources available. Please provide either a FAQ file or URL.")

# Main content area
if st.session_state.context_agent and st.session_state.qa_agent:
    # Query input
    query = st.text_input("Enter your question about IISc M.Mgt program",
                         placeholder="e.g., What is the eligibility criteria?")
    
    if query:
        with st.spinner("Generating answer..."):
            # Get answer
            result = st.session_state.qa_agent.process_query(
                query, 
                st.session_state.context_agent,
                top_k=top_k
            )
            
            # Display answer
            st.markdown("### Answer")
            st.write(result["answer"])
            
            # Display context
            with st.expander("Show Context"):
                st.markdown("### Referenced Context")
                st.write(result["context"])
            
            # Display metadata
            with st.expander("Show Details"):
                st.markdown("### Response Details")
                st.json({
                    "source_type": result["source_type"],
                    "thought_process": result["thought_process"],
                    "metadata": result["metadata"]
                })
else:
    st.info("Please initialize the system using the sidebar controls first.")
