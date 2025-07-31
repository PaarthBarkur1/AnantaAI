from faq_data import FAQ_DATA
from sentence_transformers import SentenceTransformer
from qna import (
    ContextAgent,
    JSONContentSource,
    WebContentSource,
    QAAgent,
    QAConfig,
)
import streamlit as st
import torch
import time
import logging
import re
import unicodedata
import json
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import uuid

# --- project imports ---------------------------------------------------------
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))


# --- enhanced logging -------------------------------------------------------
# Configure root logger to only show WARNING and above
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s - %(message)s"
)

# Configure our app's logger to show INFO and above
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress common warning messages
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pytorch_pretrained_bert").setLevel(logging.ERROR)
logging.getLogger("pytorch_transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

st.set_page_config(
    page_title="IISc M.Mgt QA System",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Clean, minimal CSS with consistent dark mode
st.markdown("""
<style>
    .main > div {
        padding: 2rem 1rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .answer-box {
        background: #1f2937;
        color: #f9fafb;
        border-left: 4px solid #60a5fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        white-space: pre-wrap;  /* Preserve newlines in FAQ answers */
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 1rem;
    }
    
    .status-ready {
        color: #10b981;
        font-weight: 500;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: 500;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: 500;
    }
    
    .metric-row {
        display: flex;
        gap: 2rem;
        margin: 1rem 0;
        font-size: 14px;
        color: #6b7280;
    }
    
    .diagnostic-info {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 12px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- constants ---------------------------------------------------------------
CONFIG = {
    "faq_file": "context.json",
    "sources_file": "sources.json",  # Added sources.json configuration
    "max_length": 1024,
    "max_new_tokens": 150,
    "temperature": 0.3,
    "context_window": 800,
    "top_k": 3,
    "min_confidence": 0.1,
}

# --- session state -----------------------------------------------------------


def init_state():
    defaults = dict(
        context_agent=None,
        qa_agent=None,
        embedder=None,
        system_initialized=False,
        query_history=deque(maxlen=20),
        last_query_result=None,
        suggestions=[],
        current_query="",
        initialization_logs=[],
        show_diagnostics=False,
        processing_query=False,
        query_counter=0,
        current_answer=None,  # Store current answer for consistent display
        last_processed_query="",  # Track what query was processed
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# --- utilities ---------------------------------------------------------------


def log_step(message: str, success: bool = True):
    """Log initialization steps for debugging"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status = "✓" if success else "✗"
    log_entry = f"[{timestamp}] {status} {message}"
    st.session_state.initialization_logs.append(log_entry)

    # Only log important information to terminal
    if not success:
        logger.error(message)
    elif "failed" in message.lower() or "error" in message.lower():
        logger.warning(message)
    elif "completed" in message.lower() or "initialized" in message.lower():
        logger.info(message)


def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = "".join(c for c in txt if unicodedata.category(c)[0] != "C")
    txt = txt.replace("�", "")
    txt = re.sub(r"[^\w\s\.,!?;:()\-\'\"]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    if txt and txt[-1] not in ".!?":
        txt += "."
    return txt


def extract_suggestions(text: str) -> List[str]:
    """Extract follow-up question suggestions from answer"""
    suggestions = []

    if "eligibility" in text.lower():
        suggestions.append("What documents are required for admission?")
    if "curriculum" in text.lower():
        suggestions.append("How long is the program duration?")
    if "admission" in text.lower():
        suggestions.append("What is the application deadline?")
    if "fee" in text.lower() or "cost" in text.lower():
        suggestions.append("Are there scholarship opportunities?")
    if "placement" in text.lower() or "job" in text.lower():
        suggestions.append("What is the average placement package?")

    return suggestions[:3]


def process_query(query: str) -> Dict[str, Any]:
    """Process a query and return results"""
    if not query or st.session_state.processing_query:
        return {"error": "Invalid query or processing in progress"}

    st.session_state.processing_query = True
    start = time.time()

    try:
        result = st.session_state.qa_agent.process_query(
            query,
            st.session_state.context_agent,
            top_k=CONFIG["top_k"],
        )

        elapsed = time.time() - start

        # Add to history
        st.session_state.query_history.appendleft(
            (query, datetime.now().strftime("%H:%M"), True, elapsed)
        )

        # Store result for display
        result['_processing_time'] = elapsed
        result['_query'] = query
        st.session_state.last_query_result = result
        st.session_state.last_processed_query = query

        # Process answer for display
        answer = clean_text(result.get("answer", ""))
        if not answer:
            answer = "I couldn't generate a proper response. Please try rephrasing your question."

        # Store processed answer
        st.session_state.current_answer = {
            'text': answer,
            'query': query,
            'processing_time': elapsed,
            'confidence': result.get("confidence", 0),
            'sources_count': len(result.get("metadata", {}).get("sources_used", [])),
            'sources': result.get("metadata", {}).get("sources_used", []),
            'result': result
        }

        # Generate suggestions
        new_suggestions = extract_suggestions(answer)
        if new_suggestions:
            st.session_state.suggestions = new_suggestions

        st.session_state.processing_query = False
        return result

    except Exception as exc:
        elapsed = time.time() - start
        logger.error(f"Query processing failed: {exc}")

        # Add failed query to history
        st.session_state.query_history.appendleft(
            (query, datetime.now().strftime("%H:%M"), False, elapsed)
        )

        st.session_state.processing_query = False
        return {"error": str(exc)}


@st.cache_resource(show_spinner=False)
def get_embedder() -> Optional[SentenceTransformer]:
    try:
        log_step("Loading SentenceTransformer model...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        log_step("SentenceTransformer model loaded successfully")
        return embedder
    except Exception as exc:
        log_step(f"Failed to load SentenceTransformer: {exc}", False)
        return None


@st.cache_data(show_spinner=False)
def build_index(_agent: ContextAgent, _emb: SentenceTransformer) -> Tuple[bool, str]:
    try:
        log_step("Starting semantic index build...")

        if not _agent:
            log_step("Context agent is None", False)
            return False, "Context agent is None"

        if not _emb:
            log_step("Embedder is None", False)
            return False, "Embedder is None"

        # Check if agent has data
        faq_data = _agent.get_faq_data()
        if not faq_data:
            log_step("No FAQ data available in context agent", False)
            return False, "No FAQ data available"

        log_step(f"Found {len(faq_data)} FAQ entries")

        # Build the index
        _agent.build_semantic_search_index(_emb)

        # Verify index was built
        if not hasattr(_agent, 'index') or _agent.index is None:
            log_step("Index attribute not created or is None", False)
            return False, "Index not created"

        if not hasattr(_agent, 'embedder') or _agent.embedder is None:
            log_step("Embedder not assigned to agent", False)
            return False, "Embedder not assigned"

        log_step("Semantic index built successfully")
        return True, "Success"

    except Exception as exc:
        log_step(f"Index building failed: {exc}", False)
        return False, str(exc)


def qa_config() -> QAConfig:
    return QAConfig(
        max_length=CONFIG["max_length"],
        max_new_tokens=CONFIG["max_new_tokens"],
        temperature=CONFIG["temperature"],
        top_p=0.95,
        context_window=CONFIG["context_window"],
        min_confidence=CONFIG["min_confidence"],
    )


def validate_system() -> Tuple[bool, str]:
    a, q, e = (st.session_state.context_agent,
               st.session_state.qa_agent,
               st.session_state.embedder)

    if not a:
        return False, "Context agent missing"
    if not q:
        return False, "QA agent missing"
    if not e:
        return False, "Embedder missing"
    if not getattr(a, "index", None):
        return False, "Semantic index not built"
    if not a.get_faq_data():
        return False, "No FAQ data loaded"

    return True, "System ready"

# --- initialization ----------------------------------------------------------


def initialize():
    try:
        st.session_state.initialization_logs = []
        log_step("Starting system initialization...")

        cfg = qa_config()
        agent = ContextAgent(cfg)
        log_step("ContextAgent created successfully")

        sources_added = 0

        # Add JSON FAQ source
        try:
            agent.add_source(JSONContentSource(CONFIG["faq_file"]))
            log_step(f"JSON source added: {CONFIG['faq_file']}")
            sources_added += 1
        except Exception as e:
            log_step(f"JSON source failed: {e}", False)

        # Load and add all web sources from sources.json
        try:
            with open(CONFIG["sources_file"], 'r', encoding='utf-8') as f:
                web_sources = json.load(f)
            log_step(
                f"Found {len(web_sources)} web sources in {CONFIG['sources_file']}")

            for source in web_sources:
                try:
                    web_source = WebContentSource(source["url"])
                    agent.add_source(web_source)
                    log_step(
                        f"Web source added: {source['name']} ({source['url']})")
                    sources_added += 1
                except Exception as e:
                    log_step(
                        f"Failed to add web source {source.get('name', 'Unknown')}: {e}", False)
                    continue

        except Exception as e:
            log_step(
                f"Failed to load web sources from {CONFIG['sources_file']}: {e}", False)

        if sources_added == 0:
            raise RuntimeError("No data sources available")

        faq_data = agent.get_faq_data()
        if not faq_data:
            raise RuntimeError("No FAQ data available")

        log_step(f"FAQ data loaded: {len(faq_data)} entries")

        emb = get_embedder()
        if not emb:
            raise RuntimeError("Embedding model failed to load")

        ok, msg = build_index(agent, emb)
        if not ok:
            raise RuntimeError(f"Index building failed: {msg}")

        device_id = 0 if torch.cuda.is_available() else -1
        qa_agent = QAAgent(device=device_id, config=cfg)
        log_step(
            f"QA agent created (device: {'GPU' if device_id >= 0 else 'CPU'})")

        st.session_state.context_agent = agent
        st.session_state.qa_agent = qa_agent
        st.session_state.embedder = emb
        st.session_state.system_initialized = True

        log_step("System initialization completed successfully")

    except Exception as exc:
        log_step(f"System initialization failed: {exc}", False)


# --- system status and initialization ---------------------------------------
if not st.session_state.system_initialized:
    initialize()

# --- main interface ----------------------------------------------------------
st.title("IISc M.Mgt QA Assistant")

ready, status_msg = validate_system()

# Enhanced status display with diagnostics
if ready:
    data_count = len(st.session_state.context_agent.get_faq_data())
    st.markdown(f'<p class="status-ready">● {status_msg} • {data_count} entries loaded</p>',
                unsafe_allow_html=True)
else:
    st.markdown(
        f'<p class="status-error">● {status_msg}</p>', unsafe_allow_html=True)

    # Show diagnostics for failed initialization
    st.markdown("### System Diagnostics")

    if st.session_state.initialization_logs:
        st.markdown("**Initialization Log:**")
        logs_text = "\n".join(st.session_state.initialization_logs)
        st.markdown(
            f'<div class="diagnostic-info">{logs_text}</div>', unsafe_allow_html=True)

    components = {
        "Context Agent": st.session_state.context_agent is not None,
        "QA Agent": st.session_state.qa_agent is not None,
        "Embedder": st.session_state.embedder is not None,
        "Semantic Index": bool(getattr(st.session_state.context_agent, 'index', None)),
        "FAQ Data": bool(st.session_state.context_agent and st.session_state.context_agent.get_faq_data()),
        "CUDA Available": torch.cuda.is_available(),
    }

    for component, status in components.items():
        icon = "✓" if status else "✗"
        color = "status-ready" if status else "status-error"
        st.markdown(
            f'<p class="{color}">{icon} {component}</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reinitialize System", type="primary"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state.system_initialized = False
            st.rerun()

    with col2:
        if st.button("Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Cache cleared!")

# --- main functionality (only if system is ready) -------------------------
if ready:
    # FAQ Section
    st.markdown("### Popular Questions")
    tabs = st.tabs(list(FAQ_DATA.keys()))

    for tab, (category, qas) in zip(tabs, FAQ_DATA.items()):
        with tab:
            for i, (question, answer) in enumerate(qas.items()):
                if st.button(question, key=f"faq_{category}_{i}", use_container_width=True):
                    # Display the trusted FAQ answer directly, bypass model
                    st.session_state.current_answer = {
                        'text': answer.strip(),
                        'query': question,
                        'processing_time': 0,
                        'confidence': None,
                        'sources_count': 0,
                        'sources': [],
                        'result': None
                    }
                    st.session_state.suggestions = []
                    st.session_state.last_processed_query = question
                    st.rerun()

    st.markdown("---")
    st.markdown("### Ask Your Question")

    # Show suggestions if available
    if st.session_state.suggestions:
        st.markdown("**Related questions:**")
        cols = st.columns(len(st.session_state.suggestions))
        for i, suggestion in enumerate(st.session_state.suggestions):
            # Use unique key with counter to avoid duplicates
            unique_key = f"suggest_{i}_{st.session_state.query_counter}"
            with cols[i]:
                if st.button(suggestion, key=unique_key, help="Click to search this question"):
                    st.session_state.query_counter += 1  # Increment counter
                    with st.spinner("Searching for answer..."):
                        result = process_query(suggestion)
                    if result and not result.get("error"):
                        st.rerun()

    # Main query input
    query = st.text_input(
        "",
        value=st.session_state.current_query,
        max_chars=500,
        placeholder="Ask about eligibility, curriculum, admissions, placements...",
        label_visibility="collapsed",
        key="main_query_input"
    )

    # Process manual query
    if query and len(query.strip()) > 3:
        if st.button("Search", type="primary", use_container_width=True):
            st.session_state.current_query = ""  # Clear input
            with st.spinner("Searching for answer..."):
                result = process_query(query)
            if result and not result.get("error"):
                st.rerun()
            elif result and result.get("error"):
                st.error(f"Query failed: {result['error']}")

    # CONSISTENT ANSWER PLACEMENT - Always below the chatbox
    if st.session_state.current_answer:
        answer_data = st.session_state.current_answer

        st.markdown("---")
        st.markdown("### Answer")

        # Always dark mode — no toggle
        st.markdown(
            f'<div class="answer-box">{answer_data["text"]}</div>', unsafe_allow_html=True)

        # Metrics row (skip if confidence is None — i.e., FAQ direct answers)
        if answer_data.get('confidence') is not None:
            confidence_display = (
                f"{answer_data['confidence']:.1%}" if answer_data['confidence'] is not None else "N/A")
            metrics_html = f"""
                <div class="metric-row">
                    <span>⏱ {answer_data["processing_time"]:.2f}s</span>
                    <span>📊 {confidence_display} confidence</span>
                    <span>📚 {answer_data["sources_count"]} sources</span>
                </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)

        # Sources (collapsible)
        if answer_data["sources"]:
            with st.expander("View Sources"):
                for i, source in enumerate(answer_data["sources"], 1):
                    st.write(f"**{i}.** {source.get('source', 'Unknown')}")
                    if source.get("url"):
                        st.write(f"🔗 [View]({source['url']})")
                    if source.get("confidence"):
                        st.write(f"Confidence: {source['confidence']:.1%}")
                    st.divider()

    # Recent queries with unique keys
    if st.session_state.query_history:
        with st.expander("Recent queries"):
            for i, (q, t, success, dur) in enumerate(list(st.session_state.query_history)[:5]):
                status = "✓" if success else "✗"
                unique_key = f"recent_{i}_{len(st.session_state.query_history)}"
                if st.button(f"{status} {q[:50]}...", key=unique_key,
                             help=f"{t} • {dur:.1f}s" if success else t):
                    with st.spinner("Searching for answer..."):
                        result = process_query(q)
                    if result and not result.get("error"):
                        st.rerun()

st.markdown("---")
st.caption("AI-powered assistant for IISc Master of Management program")
