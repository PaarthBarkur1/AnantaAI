import os
import json
import uvicorn
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qna import ContextAgent, QAAgent, QAConfig, JSONContentSource, WebContentSource


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

# Initialize QA system
config = QAConfig()
context_agent = ContextAgent(config)

# Add data sources
try:
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Add JSON FAQ source
    CONTEXT_PATH = os.path.join(script_dir, "context.json")
    context_agent.add_source(JSONContentSource(CONTEXT_PATH))

    # Add web sources
    SOURCES_PATH = os.path.join(script_dir, "sources.json")
    with open(SOURCES_PATH, 'r', encoding='utf-8') as f:
        web_sources = json.load(f)

    for source in web_sources:
        try:
            web_source = WebContentSource(source["url"])
            context_agent.add_source(web_source)
        except Exception as e:
            print(
                f"Failed to add web source {source.get('name', 'Unknown')}: {e}")
            continue

except Exception as e:
    print(f"Error loading data sources: {e}")

# Build semantic index
embedder = SentenceTransformer("all-MiniLM-L6-v2")
context_agent.build_semantic_search_index(embedder)
qa_agent = QAAgent(device=-1, config=config)  # Use CPU by default


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
