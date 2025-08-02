from .embedding_config import EmbeddingModelManager
import os
import json
import uvicorn
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from .qna import ContextAgent, QAAgent, QAConfig, JSONContentSource, WebContentSource
from .faq_data import FAQ_DATA
from .embedding_config import EmbeddingModelManager


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


app = FastAPI(title="IISc M.Mgt QA API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    # For development only, configure properly in production
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for QA system (will be initialized on startup)
config = None
context_agent = None
qa_agent = None
embedder = None

# Flag to prevent duplicate initialization
_initialized = False


@app.on_event("startup")
async def startup_event():
    """Initialize the QA system on startup to prevent duplicate initialization"""
    global config, context_agent, qa_agent, embedder, _initialized

    if _initialized:
        print("⚠️ QA system already initialized, skipping duplicate initialization")
        return

    print("🚀 Initializing AnantaAI QA system...")

    try:
        # Initialize configuration
        config = QAConfig()

        # Check for environment variable to disable AI generation
        if os.getenv('DISABLE_AI_GENERATION', 'false').lower() == 'true':
            config.use_ai_generation = False
            print("⚠️  AI generation disabled via environment variable")

        # Initialize context agent
        context_agent = ContextAgent(config)

        # Add data sources
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Add JSON FAQ source
        CONTEXT_PATH = os.path.join(script_dir, "context.json")
        context_agent.add_source(JSONContentSource(CONTEXT_PATH))

        # Add structured FAQ data from faq_data.py
        print("Loading structured FAQ data...")
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

        print(
            f"Loaded {faq_count} FAQ entries from {len(FAQ_DATA)} categories")

        # Load web sources during startup for complete functionality
        print("🌐 Loading web sources for complete system functionality...")

        # Load web sources (this may take a moment but provides full functionality)
        try:
            SOURCES_PATH = os.path.join(script_dir, "sources.json")
            with open(SOURCES_PATH, 'r', encoding='utf-8') as f:
                web_sources = json.load(f)

            print(f"📥 Loading {len(web_sources)} web sources...")
            successful_sources = 0
            failed_sources = 0

            for i, source in enumerate(web_sources, 1):
                source_name = source.get('name', 'Unknown')
                source_url = source.get('url', 'Unknown URL')

                try:
                    print(
                        f"   [{i}/{len(web_sources)}] Loading: {source_name}")
                    web_source = WebContentSource(source_url)
                    context_agent.add_source(web_source)
                    print(f"   ✅ Successfully added: {source_name}")
                    successful_sources += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to add {source_name}: {e}")
                    failed_sources += 1
                    continue

            print(
                f"📊 Web sources summary: {successful_sources} successful, {failed_sources} failed")

        except Exception as e:
            print(f"❌ Error loading web sources configuration: {e}")
            print("   Continuing with FAQ data only")

        # Initialize embedding model and build index
        await initialize_embedding_system()

        # Initialize QA agent
        await initialize_qa_agent()

        # Mark as initialized
        _initialized = True
        print("✅ AnantaAI QA system initialization complete!")

    except Exception as e:
        print(f"❌ Error during QA system initialization: {e}")
        raise


async def initialize_embedding_system():
    """Initialize the embedding system and build semantic index"""
    global embedder, context_agent

    print("Loading optimized sentence transformer model...")

    # Select and load the optimal model
    optimal_model = select_optimal_embedding_model()

    try:
        embedder = SentenceTransformer(optimal_model)

        # Move to GPU with error handling
        if torch.cuda.is_available():
            try:
                embedder = embedder.to('cuda')
                print("✅ Successfully moved sentence transformer to GPU")
            except Exception as e:
                print(f"⚠️ Failed to move to GPU: {e}")
                print("   Continuing with CPU processing")
                embedder = embedder.to('cpu')
        else:
            print("Using CPU for sentence transformer")

    except Exception as e:
        print(f"⚠️ Failed to load optimal model {optimal_model}: {e}")
        print("   Falling back to lightweight model")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        if torch.cuda.is_available():
            try:
                embedder = embedder.to('cuda')
            except:
                embedder = embedder.to('cpu')

    # Build semantic index
    context_agent.build_semantic_search_index(embedder)


async def initialize_qa_agent():
    """Initialize the QA agent"""
    global qa_agent, config

    # Initialize QA agent with auto-detected device
    device = get_best_device()
    print(
        f"Initializing QA agent on device: {'GPU' if device >= 0 else 'CPU'}")
    qa_agent = QAAgent(device=device, config=config)


# Auto-detect best device for models
def get_best_device():
    """Auto-detect the best available device"""
    if torch.cuda.is_available():
        device = 0  # Use first GPU
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("No GPU detected, using CPU")
        return -1


def select_optimal_embedding_model():
    """Select the best embedding model considering GPU constraints and performance needs"""

    # Check GPU availability and memory
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        try:
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)  # GB
            print(f"GPU detected with {gpu_memory:.1f}GB memory")

            # Conservative model selection based on user's previous GPU crash with large models
            if gpu_memory >= 8.0:
                # Use medium-sized model for better performance while being safe
                model_name = "sentence-transformers/all-mpnet-base-v2"
                print("Selected: all-mpnet-base-v2 (balanced performance and safety)")
            elif gpu_memory >= 4.0:
                # Use smaller but still good model
                model_name = "sentence-transformers/all-MiniLM-L12-v2"
                print("Selected: all-MiniLM-L12-v2 (good performance, moderate memory)")
            else:
                # Use lightweight model for limited GPU memory
                model_name = "all-MiniLM-L6-v2"
                print("Selected: all-MiniLM-L6-v2 (lightweight, safe for limited GPU)")
        except Exception as e:
            print(f"GPU memory check failed: {e}, using safe default")
            model_name = "all-MiniLM-L6-v2"
    else:
        # CPU-only: use efficient model
        model_name = "sentence-transformers/all-MiniLM-L12-v2"
        print("Selected: all-MiniLM-L12-v2 (CPU optimized)")

    return model_name


# All initialization is now handled in the startup event


class Query(BaseModel):
    text: str
    max_results: Optional[int] = 3


class QAResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    processing_time: float


@app.post("/api/query", response_model=QAResponse)
async def process_query(query: Query):
    # Check if system is initialized
    if not _initialized or qa_agent is None or context_agent is None:
        raise HTTPException(
            status_code=503,
            detail="QA system is still initializing. Please wait a moment and try again."
        )

    try:
        result = qa_agent.process_query(
            query.text, context_agent, top_k=query.max_results)

        # Convert numpy types to Python types for JSON serialization
        response_data = {
            "answer": result["answer"],
            "confidence": convert_numpy_types(result.get("confidence", 0.0)),
            "sources": convert_numpy_types(result.get("metadata", {}).get("sources_used", [])),
            "processing_time": convert_numpy_types(result.get("_processing_time", 0.0))
        }

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint to monitor system status"""
    return {
        "status": "ready" if _initialized else "initializing",
        "initialized": _initialized,
        "components": {
            "context_agent": context_agent is not None,
            "qa_agent": qa_agent is not None,
            "embedder": embedder is not None
        }
    }


@app.get("/api/categories")
async def get_categories():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sources_path = os.path.join(script_dir, "sources.json")
        with open(sources_path, "r", encoding="utf-8") as f:
            sources = json.load(f)
            categories = set()
            for source in sources:
                categories.update(source["category"])
            return {"categories": sorted(list(categories))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
