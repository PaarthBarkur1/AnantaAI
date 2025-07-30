from sentence_transformers import SentenceTransformer
from AnantaAI.backend.qna import ContextAgent, QAAgent, QAConfig, JSONContentSource, WebContentSource
import json
import uvicorn
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import os

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
    # Add JSON FAQ source
    import os
    CONTEXT_PATH = os.path.join(os.path.dirname(__file__), "context.json")
    context_agent.add_source(JSONContentSource(CONTEXT_PATH))

    # Add web sources
    SOURCES_PATH = os.path.join(os.path.dirname(__file__), "sources.json")
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
        return {
            "answer": result["answer"],
            "confidence": result.get("confidence", 0.0),
            "sources": result.get("metadata", {}).get("sources_used", []),
            "processing_time": result.get("_processing_time", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories")
async def get_categories():
    try:
        with open("sources.json", "r", encoding="utf-8") as f:
            sources = json.load(f)
            categories = set()
            for source in sources:
                categories.update(source["category"])
            return {"categories": sorted(list(categories))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("AnantaAI.backend.main:app",
                host="0.0.0.0", port=8000, reload=True)
