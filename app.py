import streamlit as st
import torch
import time
import logging
from typing import Dict, Any, Optional
import hashlib
import re
import unicodedata

# Import your enhanced QNA module
from qna import ContextAgent, JSONContentSource, WebContentSource, QAAgent, SentenceTransformer, QAConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="IISc M.Mgt QA System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with dark theme answer container and better styling
st.markdown("""
<style>
    .stAlert > div {
        padding: 0.5rem 1rem;
    }
    .main-header {
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-ready {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-not-ready {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .status-error {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Dark Answer Container - Fixed for readability */
    .answer-container {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        padding: 1.5rem;
        border-left: 4px solid #4CAF50;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    
    /* Light Answer Container Alternative */
    .answer-container-light {
        background-color: #f8f9fa !important;
        color: #212529 !important;
        padding: 1.5rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    /* Query history styling */
    .query-history-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        background-color: #f8f9fa;
        border-left: 3px solid #007bff;
    }
    
    /* Error message styling */
    .error-detail {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    
    /* Loading indicator */
    .loading-text {
        text-align: center;
        color: #6c757d;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with comprehensive defaults


def init_session_state():
    """Initialize session state with all required variables"""
    defaults = {
        'context_agent': None,
        'qa_agent': None,
        'embedder': None,
        'system_initialized': False,
        'system_validated': False,
        'last_config_hash': None,
        'query_history': [],
        'performance_stats': {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'successful_queries': 0,
            'failed_queries': 0
        },
        'initialization_error': None,
        'last_query_result': None,
        'answer_theme': 'dark',  # New: track answer theme preference
        'debug_mode': False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# Text cleaning utilities


def clean_generated_text(text: str) -> str:
    """Clean and validate generated text to prevent garbled output"""
    if not text:
        return ""

    # Remove control characters and invalid unicode
    text = ''.join(
        char for char in text if unicodedata.category(char)[0] != 'C')

    # Replace replacement characters
    text = text.replace('�', '')

    # Remove excessive special characters and symbols (but keep punctuation)
    text = re.sub(r'[^\w\s\.,!?;:()\-\'"]+', ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove repeated characters (more than 3 in a row)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # Clean up quotes and special characters
    text = text.replace('""', '"').replace("''", "'")

    # Ensure the text ends properly
    text = text.strip()
    if text and not text.endswith(('.', '!', '?', ':')):
        text += '.'

    return text

# Caching functions with better error handling


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder() -> Optional[SentenceTransformer]:
    """Load and cache the sentence transformer model."""
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Successfully loaded embedding model")
        return embedder
    except Exception as e:
        logger.error(f"Failed to load embedder: {e}")
        st.error(
            "Failed to load embedding model. Please check your internet connection.")
        return None


@st.cache_data(show_spinner="Building semantic search index...")
def build_search_index(_context_agent, _embedder) -> tuple[bool, str]:
    """Build and cache the semantic search index with detailed error reporting."""
    try:
        if not _context_agent:
            return False, "Context agent is None"

        if not _embedder:
            return False, "Embedder is None"

        faq_data = _context_agent.get_faq_data()
        if not faq_data:
            return False, "No FAQ data available"

        logger.info(f"Building index for {len(faq_data)} entries")

        # Call the actual method
        _context_agent.build_semantic_search_index(_embedder)

        # Verify the index was built
        if not hasattr(_context_agent, 'index') or _context_agent.index is None:
            return False, "Index was not created"

        if not hasattr(_context_agent, 'embedder') or _context_agent.embedder is None:
            return False, "Embedder was not assigned"

        logger.info("Successfully built semantic search index")
        return True, "Success"

    except Exception as e:
        error_msg = f"Failed to build search index: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def get_config_hash(faq_file: str, url: str, device: str, **kwargs) -> str:
    """Generate a hash of the current configuration."""
    content = f"{faq_file}_{url}_{device}"
    for key, value in kwargs.items():
        content += f"_{key}_{value}"
    return hashlib.md5(content.encode()).hexdigest()


def create_qa_config(args: Dict[str, Any]) -> QAConfig:
    """Create QA configuration from UI inputs"""
    return QAConfig(
        max_length=args.get('max_length', 1024),
        max_new_tokens=args.get('max_new_tokens', 256),
        temperature=args.get('temperature', 0.7),
        top_p=args.get('top_p', 0.95),
        context_window=args.get('context_window', 800),
        min_confidence=args.get('min_confidence', 0.1)
    )

# System validation functions


def validate_system_ready(during_init: bool = False) -> tuple[bool, str]:
    """Validate that the system is properly initialized and ready"""

    # During initialization, use different validation logic
    if during_init:
        # Just check if basic components exist
        context_agent = st.session_state.get('context_agent')
        qa_agent = st.session_state.get('qa_agent')

        if not context_agent:
            return False, "Context agent not created"

        if not qa_agent:
            return False, "QA agent not created"

        # Check if context agent has required attributes
        if not hasattr(context_agent, 'index') or context_agent.index is None:
            return False, "Semantic index not built"

        if not hasattr(context_agent, 'embedder') or context_agent.embedder is None:
            return False, "Embedder not available"

        return True, "System components ready"

    # Normal validation after initialization
    if not st.session_state.get('system_initialized', False):
        return False, "System not initialized"

    context_agent = st.session_state.get('context_agent')
    if not context_agent:
        return False, "Context agent not available"

    if not st.session_state.get('qa_agent'):
        return False, "QA agent not available"

    # Check if the context agent has the is_ready method
    if hasattr(context_agent, 'is_ready'):
        if not context_agent.is_ready():
            return False, "Context agent reports not ready"
    else:
        # Fallback checks if is_ready method doesn't exist
        if not hasattr(context_agent, 'index') or context_agent.index is None:
            return False, "Semantic index not built"

        if not hasattr(context_agent, 'embedder') or context_agent.embedder is None:
            return False, "Embedder not available"

        if not context_agent.get_faq_data():
            return False, "No data loaded"

    return True, "System ready"


def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    if st.session_state.context_agent:
        try:
            status = st.session_state.context_agent.get_status()
            status['system_initialized'] = st.session_state.system_initialized
            status['system_validated'] = st.session_state.system_validated
            return status
        except Exception as e:
            return {'error': str(e), 'system_initialized': False}
    return {'system_initialized': False}

# Performance tracking


def update_performance_stats(processing_time: float, success: bool = True):
    """Update performance statistics"""
    stats = st.session_state.performance_stats
    stats['total_queries'] += 1

    if success:
        stats['successful_queries'] += 1
        stats['avg_response_time'] = (
            (stats['avg_response_time'] *
             (stats['successful_queries'] - 1) + processing_time)
            / stats['successful_queries']
        )
    else:
        stats['failed_queries'] += 1

# Enhanced answer processing function


def process_and_display_answer(result: Dict[str, Any], processing_time: float):
    """Process and display the answer with proper formatting"""
    # Clean the answer text
    raw_answer = result.get('answer', 'No answer generated')
    cleaned_answer = clean_generated_text(raw_answer)

    # Fallback if cleaning didn't work
    if not cleaned_answer or len(cleaned_answer) < 10:
        cleaned_answer = (
            "I apologize, but I encountered an issue generating a readable response. "
            "The system found relevant information, but couldn't format it properly. "
            "Please try rephrasing your question or reinitialize the system."
        )

    # Display answer with theme selection
    st.markdown("### 💡 Answer")

    # Theme toggle
    col1, col2 = st.columns([4, 1])
    with col2:
        dark_theme = st.checkbox(
            "🌙 Dark Theme",
            value=st.session_state.get('answer_theme', 'dark') == 'dark',
            help="Toggle answer background theme"
        )
        st.session_state.answer_theme = 'dark' if dark_theme else 'light'

    # Display answer with chosen theme
    if dark_theme:
        st.markdown(f"""
        <div class="answer-container">
{cleaned_answer}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="answer-container-light">
{cleaned_answer}
        </div>
        """, unsafe_allow_html=True)

    return cleaned_answer


# Header with enhanced status indicator
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("IISc M.Mgt QA System 🎓")

# Enhanced system status indicator
system_status = get_system_status()
is_ready, status_message = validate_system_ready()

if is_ready:
    status_class = "status-ready"
    status_text = "✅ System Ready & Validated"
    data_count = len(st.session_state.context_agent.get_faq_data()
                     ) if st.session_state.context_agent else 0
    status_text += f" ({data_count} entries loaded)"
elif st.session_state.initialization_error:
    status_class = "status-error"
    status_text = f"⚠️ Initialization Error: {st.session_state.initialization_error}"
else:
    status_class = "status-not-ready"
    status_text = f"❌ {status_message}"

st.markdown(f"""
<div class="status-indicator {status_class}">
    <strong>{status_text}</strong>
</div>
""", unsafe_allow_html=True)

st.markdown("""
This system answers questions about the IISc M.Mgt program using FAQ data and web content with advanced semantic search.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Enhanced sidebar configuration
# Enhanced sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Performance metrics display
    if st.session_state.performance_stats['total_queries'] > 0:
        st.subheader("📊 Performance Metrics")
        stats = st.session_state.performance_stats

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", stats['total_queries'])
            st.metric(
                "Success Rate", f"{(stats['successful_queries']/stats['total_queries']*100):.1f}%")
        with col2:
            st.metric("Avg Response", f"{stats['avg_response_time']:.2f}s")
            st.metric("Failed Queries", stats['failed_queries'])
        st.divider()

    # Data sources section
    st.subheader("📁 Data Sources")

    faq_file = st.text_input(
        "FAQ JSON file path",
        value="context.json",
        help="Path to your FAQ JSON file containing question-answer pairs",
        placeholder="e.g., context.json"
    )

    url = st.text_input(
        "URL to scrape",
        help="Enter a URL to scrape content from for additional context",
        placeholder="https://example.com/iisc-program-info"
    )

    # Validate data sources
    sources_available = bool(faq_file.strip() or url.strip())
    if not sources_available:
        st.warning("⚠️ At least one data source is required")

    # Model settings section
    st.subheader("🤖 Model Settings")

    device_options = ["GPU", "CPU"]
    default_device = 0 if torch.cuda.is_available() else 1

    device = st.radio(
        "Processing Device",
        options=device_options,
        index=default_device,
        help="GPU provides faster processing if available"
    )

    if device == "GPU" and not torch.cuda.is_available():
        st.warning("⚠️ GPU not available, will use CPU instead")
        device = "CPU"

    # Generation Settings - NEW ADDITION for fixing text generation
    st.subheader("🎯 Generation Settings")

    generation_mode = st.radio(
        "Answer Generation Mode",
        options=["AI Generation", "Context Extraction", "Hybrid"],
        index=2,  # Default to Hybrid
        help="Choose how answers are generated"
    )

    # Store in session state
    st.session_state.generation_mode = generation_mode

    # Advanced model parameters
    with st.expander("🔧 Advanced Parameters"):
        max_length = st.slider(
            "Model Max Length",
            min_value=512,
            max_value=2048,
            value=1024,
            step=128,
            help="Maximum total tokens the model can handle"
        )

        max_new_tokens = st.slider(
            "Max New Tokens",
            min_value=50,
            max_value=512,
            value=150,  # Reduced for better stability
            step=32,
            help="Maximum tokens for generated answers"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.3,  # Lower for more focused output
            step=0.1,
            help="Controls randomness in generation (lower = more focused)"
        )

        context_window = st.slider(
            "Context Window",
            min_value=200,
            max_value=1500,
            value=800,
            step=100,
            help="Maximum tokens allocated for context"
        )

    # Search and quality settings
    st.subheader("🔍 Search Settings")

    top_k = st.slider(
        "Context retrieval count",
        min_value=1,
        max_value=15,
        value=5,
        help="Number of relevant contexts to retrieve"
    )

    min_confidence = st.slider(
        "Minimum confidence threshold",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        help="Filter out low-confidence search results"
    )

    # Configuration change detection
    config_params = {
        'max_length': max_length,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'context_window': context_window,
        'min_confidence': min_confidence,
        'generation_mode': generation_mode
    }
    current_config_hash = get_config_hash(
        faq_file, url, device, **config_params)
    config_changed = st.session_state.last_config_hash != current_config_hash

    if config_changed and st.session_state.system_initialized:
        st.info("🔄 Configuration changed. Reinitialize to apply changes.")

    # Control buttons
    st.subheader("🎛️ System Controls")

    col1, col2 = st.columns([2, 1])

    with col1:
        init_button_text = "🚀 Initialize System" if not st.session_state.system_initialized else "🔄 Reinitialize"
        init_button = st.button(
            init_button_text,
            disabled=not sources_available,
            use_container_width=True,
            type="primary"
        )

    with col2:
        if st.button("🧹", help="Clear cache and reset", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            # Reset session state
            for key in ['context_agent', 'qa_agent', 'embedder', 'system_initialized', 'system_validated']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("System reset!")
            st.rerun()

    # Emergency controls for troubleshooting
    st.subheader("🚨 Troubleshooting")

    if st.button("🔧 Force Rebuild Index", help="Clear cache and rebuild search index"):
        if st.session_state.get('context_agent') and st.session_state.get('embedder'):
            try:
                build_search_index.clear()
                with st.spinner("Rebuilding search index..."):
                    context_agent = st.session_state.context_agent
                    embedder = st.session_state.embedder
                    success, message = build_search_index(
                        context_agent, embedder)
                    if success:
                        st.success("✅ Index rebuilt successfully!")
                        st.session_state.system_validated = True
                    else:
                        st.error(f"❌ Failed to rebuild index: {message}")
            except Exception as e:
                st.error(f"❌ Error rebuilding index: {e}")
        else:
            st.warning("⚠️ System not initialized. Please initialize first.")

    # Test system button
    if st.button("🧪 Test System", help="Test the system with a simple query"):
        if st.session_state.system_initialized:
            test_query = "What is the M.Mgt program?"
            try:
                with st.spinner("Testing system..."):
                    result = st.session_state.qa_agent.process_query(
                        test_query,
                        st.session_state.context_agent,
                        top_k=3
                    )
                st.success("✅ System test completed!")
                with st.expander("Test Results"):
                    st.json(result)
            except Exception as e:
                st.error(f"❌ System test failed: {e}")
        else:
            st.warning("⚠️ Initialize system first")

    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox(
        "🐛 Debug Mode",
        value=st.session_state.get('debug_mode', False),
        help="Show detailed debug information"
    )

    # Debug information display
    if st.session_state.debug_mode:
        st.subheader("🔍 Debug Information")
        debug_info = {
            "System Initialized": st.session_state.get('system_initialized', False),
            "Context Agent": st.session_state.get('context_agent') is not None,
            "QA Agent": st.session_state.get('qa_agent') is not None,
            "Embedder": st.session_state.get('embedder') is not None,
            "CUDA Available": torch.cuda.is_available(),
            "Performance Stats": st.session_state.get('performance_stats', {}),
            "Last Config Hash": st.session_state.get('last_config_hash', 'None')
        }

        if st.session_state.get('context_agent'):
            agent = st.session_state.context_agent
            debug_info.update({
                "Has Index": hasattr(agent, 'index') and agent.index is not None,
                "Has Embedder": hasattr(agent, 'embedder') and agent.embedder is not None,
                "FAQ Data Count": len(agent.get_faq_data()) if hasattr(agent, 'get_faq_data') else 0
            })

        st.json(debug_info)

# Enhanced system initialization
if init_button and sources_available:
    # Clear any previous errors
    st.session_state.initialization_error = None

    # Reset system state first
    st.session_state.system_initialized = False
    st.session_state.system_validated = False
    st.session_state.context_agent = None
    st.session_state.qa_agent = None

    # Progress tracking with detailed steps
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Create QA configuration
        status_text.text("Creating system configuration...")
        progress_bar.progress(10)
        qa_config = create_qa_config(config_params)
        time.sleep(0.2)

        # Step 2: Initialize context agent
        status_text.text("Initializing context agent...")
        progress_bar.progress(20)
        context_agent = ContextAgent(qa_config)

        # Step 3: Load and validate data sources
        status_text.text("Loading data sources...")
        progress_bar.progress(30)

        sources_loaded = []
        total_entries = 0

        if faq_file.strip():
            try:
                faq_source = JSONContentSource(faq_file.strip())
                context_agent.add_source(faq_source)
                faq_data = faq_source.get_content()
                sources_loaded.append(f"FAQ file: {len(faq_data)} entries")
                total_entries += len(faq_data)
            except Exception as e:
                st.warning(f"Failed to load FAQ file: {e}")

        if url.strip():
            try:
                web_source = WebContentSource(url.strip())
                context_agent.add_source(web_source)
                web_data = web_source.get_content()
                sources_loaded.append(f"Web content: {len(web_data)} sections")
                total_entries += len(web_data)
            except Exception as e:
                st.warning(f"Failed to scrape URL: {e}")

        if not sources_loaded or total_entries == 0:
            raise Exception("No valid data could be loaded from any source")

        progress_bar.progress(50)

        # Step 4: Load embedding model
        status_text.text("Loading embedding model...")
        embedder = get_embedder()
        if not embedder:
            raise Exception("Failed to load embedding model")

        progress_bar.progress(70)

        # Step 5: Build semantic search index with detailed error checking
        status_text.text("Building semantic search index...")

        # Clear any cached version first
        build_search_index.clear()

        index_success, index_message = build_search_index(
            context_agent, embedder)
        if not index_success:
            raise Exception(
                f"Failed to build semantic search index: {index_message}")

        progress_bar.progress(85)

        # Step 6: Initialize QA agent
        status_text.text("Initializing QA model...")
        device_id = 0 if device == "GPU" and torch.cuda.is_available() else -1
        qa_agent = QAAgent(device=device_id, config=qa_config)

        progress_bar.progress(95)

        # Step 7: Final system setup
        status_text.text("Finalizing system setup...")

        # Update session state
        st.session_state.context_agent = context_agent
        st.session_state.qa_agent = qa_agent
        st.session_state.embedder = embedder

        # Test search functionality directly
        try:
            test_results = context_agent.search("test query", top_k=1)
            logger.info(f"Search test completed successfully")
        except Exception as e:
            raise Exception(f"Search functionality test failed: {e}")

        # Step 8: Mark as ready
        progress_bar.progress(100)
        status_text.text("System ready!")

        # Mark system as initialized
        st.session_state.system_initialized = True
        st.session_state.system_validated = True
        st.session_state.last_config_hash = current_config_hash

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Show success message with details
        success_msg = f"✅ System initialized successfully!\n\n"
        success_msg += f"📊 **Loaded Sources:**\n"
        for source_info in sources_loaded:
            success_msg += f"• {source_info}\n"
        success_msg += f"\n🎯 **Total Entries:** {total_entries}"
        success_msg += f"\n🔍 **Index Status:** {index_message}"

        st.success(success_msg)
        time.sleep(2)
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        error_msg = str(e)
        st.session_state.initialization_error = error_msg
        st.error(f"❌ Initialization failed: {error_msg}")
        logger.error(f"System initialization failed: {e}")

        # Reset system state on failure
        st.session_state.system_initialized = False
        st.session_state.system_validated = False
        st.session_state.context_agent = None
        st.session_state.qa_agent = None

# Main query interface
if st.session_state.system_initialized:
    # Validate system before showing interface
    is_ready, status_msg = validate_system_ready()

    if not is_ready:
        st.error(f"❌ System Error: {status_msg}")
        st.info("Please reinitialize the system using the sidebar controls.")
        st.stop()

    # Query interface
    st.subheader("💬 Ask Your Question")

    # Query history in expandable section
    if st.session_state.query_history:
        with st.expander(f"📚 Recent Queries ({len(st.session_state.query_history)})", expanded=False):
            for i, (query, timestamp, success) in enumerate(reversed(st.session_state.query_history[-10:])):
                status_icon = "✅" if success else "❌"
                if st.button(f"{status_icon} {query[:60]}...", key=f"history_{i}", help=f"Asked at {timestamp}"):
                    st.session_state.current_query = query

    # Main query input
    query = st.text_input(
        "Enter your question about the IISc M.Mgt program",
        placeholder="e.g., What are the eligibility criteria for the M.Mgt program?",
        value=st.session_state.get('current_query', ''),
        help="Ask any question about the IISc M.Mgt program",
        max_chars=500
    )

    # Clear current query after use
    if 'current_query' in st.session_state:
        del st.session_state.current_query

    # Query processing
    if query.strip():
        # Input validation
        if len(query.strip()) < 5:
            st.warning("⚠️ Please enter a more detailed question.")
        elif len(query.split()) > 100:
            st.warning(
                "⚠️ Question is too long. Please try a shorter, more specific question.")
        else:
            col1, col2 = st.columns([3, 1])

            with col2:
                ask_button = st.button(
                    "🔍 Ask Question",
                    type="primary",
                    use_container_width=True,
                    help="Process your question using AI"
                )

            if ask_button:
                start_time = time.time()

                try:
                    with st.spinner("🤔 Processing your question..."):
                        # Process the query
                        result = st.session_state.qa_agent.process_query(
                            query,
                            st.session_state.context_agent,
                            top_k=top_k
                        )

                        processing_time = time.time() - start_time

                        # Check for errors in result
                        if result.get('source_type') == 'error':
                            error_msg = result.get('metadata', {}).get(
                                'error', 'Unknown error')
                            if 'Semantic index not built' in error_msg:
                                st.error(
                                    "❌ Search index corrupted. Please reinitialize the system.")
                                st.session_state.system_initialized = False
                                st.stop()
                            else:
                                st.error(f"❌ Processing error: {error_msg}")
                                update_performance_stats(
                                    processing_time, success=False)
                                # Add to history as failed
                                timestamp = time.strftime("%H:%M:%S")
                                st.session_state.query_history.append(
                                    (query, timestamp, False))
                                st.stop()

                        # Successful processing
                        update_performance_stats(processing_time, success=True)

                        # Add to query history
                        timestamp = time.strftime("%H:%M:%S")
                        st.session_state.query_history.append(
                            (query, timestamp, True))
                        st.session_state.last_query_result = result

                        # Keep only last 50 queries
                        if len(st.session_state.query_history) > 50:
                            st.session_state.query_history = st.session_state.query_history[-50:]

                    # Enhanced answer display with error handling
                    cleaned_answer = process_and_display_answer(
                        result, processing_time)

                    # Metrics display
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Response Time", f"{processing_time:.2f}s")

                    with col2:
                        confidence = result.get('confidence', 0)
                        if confidence > 0:
                            conf_class = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
                            st.markdown(
                                f'<div class="confidence-{conf_class}">Confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
                        else:
                            st.metric("Confidence", "N/A")

                    with col3:
                        sources_count = len(result.get(
                            'metadata', {}).get('sources_used', []))
                        st.metric("Sources Used", sources_count)

                    with col4:
                        contexts_found = result.get(
                            'metadata', {}).get('contexts_found', 0)
                        st.metric("Contexts Found", contexts_found)

                    # Debug information if enabled
                    if st.session_state.debug_mode:
                        with st.expander("🐛 Debug Information", expanded=False):
                            debug_info = {
                                "Raw Answer Length": len(result.get('answer', '')),
                                "Cleaned Answer Length": len(cleaned_answer),
                                "Has Special Characters": bool(re.search(r'[^\w\s\.,!?;:()\-\'"]', result.get('answer', ''))),
                                "Processing Steps": {
                                    "Query Processing": f"{processing_time:.3f}s",
                                    "Search Results": len(result.get('metadata', {}).get('sources_used', [])),
                                    "Context Length": len(result.get('context', ''))
                                },
                                "Raw Answer Preview": result.get('answer', '')[:200] + "..." if len(result.get('answer', '')) > 200 else result.get('answer', '')
                            }
                            st.json(debug_info)

                    # Expandable sections for detailed information
                    with st.expander("📖 Referenced Context", expanded=False):
                        context = result.get("context", "No context available")
                        if context:
                            st.text_area(
                                "Context used for answer generation:", context, height=200, disabled=True)
                        else:
                            st.info("No context was used for this answer.")

                    with st.expander("🔍 Technical Details", expanded=False):
                        metadata = result.get("metadata", {})

                        # Sources breakdown
                        sources = metadata.get("sources_used", [])
                        if sources:
                            st.markdown("**Sources Breakdown:**")
                            for i, source in enumerate(sources, 1):
                                source_type = source.get("source", "unknown")
                                confidence = source.get("confidence", 0)
                                truncated = source.get("truncated", False)

                                st.markdown(f"**{i}. {source_type.upper()}**")
                                if confidence > 0:
                                    st.markdown(
                                        f"   - Confidence: {confidence:.1%}")
                                if truncated:
                                    st.markdown(
                                        "   - ⚠️ Content was truncated")
                                if source.get("url"):
                                    st.markdown(f"   - URL: {source['url']}")
                                if source.get("question") and not source["question"].startswith("Web Content"):
                                    st.markdown(
                                        f"   - Topic: {source['question']}")

                        # Technical metrics
                        st.markdown("**Technical Metrics:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(
                                f"- Processing Time: {processing_time:.3f}s")
                            st.markdown(
                                f"- Source Type: {result.get('source_type', 'unknown')}")
                        with col2:
                            if metadata.get("prompt_tokens"):
                                st.markdown(
                                    f"- Prompt Tokens: {metadata['prompt_tokens']}")
                            st.markdown(
                                f"- Contexts Found: {metadata.get('contexts_found', 0)}")

                    # User feedback section
                    st.markdown("### 📝 Feedback")
                    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)

                    query_id = len(st.session_state.query_history)

                    with feedback_col1:
                        if st.button("👍 Helpful", key=f"helpful_{query_id}"):
                            st.success("Thank you for your positive feedback!")

                    with feedback_col2:
                        if st.button("👎 Not Helpful", key=f"not_helpful_{query_id}"):
                            st.info(
                                "Thank you for your feedback. We'll work on improving!")

                    with feedback_col3:
                        if st.button("🔄 Ask Follow-up", key=f"followup_{query_id}"):
                            st.session_state.current_query = f"Follow-up to: {query[:50]}... "
                            st.rerun()

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Query processing failed: {e}")

                    # Handle specific errors
                    if 'Semantic index not built' in error_msg:
                        st.error(
                            "❌ Search system corrupted. Please reinitialize.")
                        st.session_state.system_initialized = False
                    elif 'token' in error_msg.lower():
                        st.error(
                            "❌ Question too complex. Try a shorter, simpler question.")
                    else:
                        st.error(f"❌ Error processing query: {error_msg}")

                    # Add to failed queries
                    update_performance_stats(
                        time.time() - start_time, success=False)
                    timestamp = time.strftime("%H:%M:%S")
                    st.session_state.query_history.append(
                        (query, timestamp, False))

                    # Show troubleshooting tips
                    with st.expander("🛠️ Troubleshooting Tips"):
                        st.markdown("""
                        **Try these solutions:**
                        - Rephrase your question to be more specific
                        - Break complex questions into smaller parts
                        - Check if the system is properly initialized
                        - Clear cache and reinitialize if problems persist
                        - Ensure your data sources are accessible
                        - Try reducing the max_new_tokens in advanced settings
                        """)

else:
    # Enhanced onboarding experience
    st.markdown("""
    ### 🚀 Getting Started
    
    Welcome to the IISc M.Mgt QA System! Follow these steps to begin:
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        **1. Configure Data Sources 📁**
        - Add a FAQ JSON file path, or
        - Provide a URL to scrape content from
        - At least one source is required
        
        **2. Adjust Settings ⚙️**
        - Choose GPU for better performance (if available)
        - Configure quality and search parameters
        - Advanced users can fine-tune model parameters
        """)

    with col2:
        st.markdown("""
        **3. Initialize System 🚀**
        - Click "Initialize System" in the sidebar
        - Wait for the system to load and validate
        - Green status indicator confirms readiness
        
        **4. Start Asking Questions 💬**
        - Type your question in the input field
        - Get AI-powered answers with source attribution
        - View detailed context and technical information
        """)

    # Sample questions with categories
    st.markdown("""
    ### 💡 Sample Questions by Category
    """)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📚 Academics", "💰 Finances", "📝 Admissions", "🎯 Career"])

    with tab1:
        st.markdown("""
        - What is the curriculum structure of the M.Mgt program?
        - How many credits are required to graduate?
        - What are the core subjects in the first semester?
        - Are there any specialization tracks available?
        """)

    with tab2:
        st.markdown("""
        - What is the fee structure for the M.Mgt program?
        - Are there any scholarships available?
        - What are the hostel charges?
        - Are there any financial aid options?
        """)

    with tab3:
        st.markdown("""
        - What are the eligibility criteria for admission?
        - What is the application process?
        - When are the application deadlines?
        - What documents are required for application?
        """)

    with tab4:
        st.markdown("""
        - What are the career prospects after graduation?
        - Which companies recruit from the program?
        - What is the average placement package?
        - Are there entrepreneurship opportunities?
        """)

# Footer with system information
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <strong>IISc M.Mgt QA System v2.0</strong><br>
        Built with Streamlit • Enhanced with AI • Powered by Semantic Search<br>
        🌙 Dark theme answers • 🔧 Advanced text processing • 🛡️ Error recovery<br>
        For technical support, enable debug mode or reinitialize the system
    </div>
    """, unsafe_allow_html=True)

# Debug information (only show if debug mode is enabled)
if st.session_state.debug_mode:
    with st.sidebar:
        st.markdown("### 🐛 Debug Information")
        debug_info = {
            "System Initialized": st.session_state.system_initialized,
            "System Validated": st.session_state.system_validated,
            "Context Agent": st.session_state.context_agent is not None,
            "QA Agent": st.session_state.qa_agent is not None,
            "Embedder": st.session_state.embedder is not None,
            "Query History Length": len(st.session_state.query_history),
            "CUDA Available": torch.cuda.is_available(),
            "Answer Theme": st.session_state.get('answer_theme', 'dark'),
            "Performance Stats": st.session_state.performance_stats
        }
        st.json(debug_info)
