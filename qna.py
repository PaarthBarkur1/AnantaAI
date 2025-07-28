import json
import argparse
from typing import List, Dict, Any
from abc import ABC, abstractmethod

import faiss
from sentence_transformers import SentenceTransformer

from transformers import pipeline

from webscrapper import WebScraper


# ---------------------------
# Content Source Abstract and Implementations
# ---------------------------

class ContentSource(ABC):
    """Abstract base class"""
    @abstractmethod
    def get_content(self) -> List[Dict[str, Any]]:
        pass


class JSONContentSource(ContentSource):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def get_content(self) -> List[Dict[str, Any]]:
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON data must be a list of QA pairs")
            return data
        except Exception as e:
            raise RuntimeError(
                f"Failed to load JSON data from {self.filepath}: {e}")


class WebContentSource(ContentSource):
    def __init__(self, url: str):
        self.url = url
        self.scraper = WebScraper()

    def get_content(self) -> List[Dict[str, Any]]:
        paragraphs = self.scraper.scrape_paragraphs(self.url)
        # Wrap paragraphs as FAQ-like entries with generic questions
        return [
            {
                "question": f"Web Content #{i + 1}",
                "answer": para,
                "metadata": {"source": "web", "url": self.url}
            }
            for i, para in enumerate(paragraphs) if para.strip()
        ]


# ---------------------------
# Context Agent with Semantic Search
# ---------------------------

class ContextAgent:
    def __init__(self):
        self.sources: List[ContentSource] = []
        self.faq_data: List[Dict[str, Any]] = []

        self.embedder = None
        self.doc_embeddings = None
        self.index = None
        self.documents = None

    def add_source(self, source: ContentSource) -> None:
        self.sources.append(source)
        self.faq_data.extend(source.get_content())

    def get_faq_data(self) -> List[Dict[str, Any]]:
        return self.faq_data

    def get_source_confidence(self, source_type: str) -> float:
        confidence_scores = {
            "json": 0.9,
            "web": 0.7,
        }
        return confidence_scores.get(source_type, 0.5)

    def build_semantic_search_index(self, embedder: SentenceTransformer) -> None:
        self.embedder = embedder
        self.documents = []
        for entry in self.faq_data:
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            source = entry.get("metadata", {}).get("source", "unknown")
            doc_text = f"Question: {question}\nAnswer: {answer}\nSource: {source}"
            self.documents.append(doc_text)
        if not self.documents:
            raise ValueError("No documents to index for semantic search.")

        self.doc_embeddings = embedder.encode(
            self.documents, convert_to_numpy=True)
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.doc_embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.embedder is None:
            raise RuntimeError(
                "Semantic index not built. Call build_semantic_search_index() first.")

        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        return [self.faq_data[idx] for idx in indices[0]]


# ---------------------------
# QA Agent - Llama 2 7B Chat
# ---------------------------

class QAAgent:
    def __init__(self, device: int = 0):
        # Check if CUDA is available and use it
        import torch
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            device = 0  # Use the first GPU
        else:
            print("CUDA not available, using CPU")
            device = -1
            
        self.generator = pipeline(
            "text-generation",
            model="facebook/opt-350m",  # Using a smaller, public model
            device=device,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )

    def process_query(self, query: str, context_agent: ContextAgent, top_k: int = 5) -> Dict[str, Any]:
        top_contexts = context_agent.search(query, top_k=top_k)
        
        # Format contexts with their sources
        formatted_contexts = []
        used_sources = []
        for ctx in top_contexts:
            source = ctx.get("metadata", {}).get("source", "unknown")
            source_url = ctx.get("metadata", {}).get("url", "")
            source_info = f"(Source: {source})"
            if source_url:
                source_info += f" from {source_url}"
            if ctx.get("question", "").startswith("Web Content #"):
                formatted_contexts.append(f"{ctx.get('answer', '')}\n{source_info}")
            else:
                formatted_contexts.append(f"Q: {ctx.get('question', '')}\nA: {ctx.get('answer', '')}\n{source_info}")
            used_sources.append({"source": source, "url": source_url, "question": ctx.get("question", "")})
        
        context_text = "\n\n".join(formatted_contexts)

        prompt = (
            f"You are an expert AI assistant specializing in IISc M.Mgt program information. "
            f"Use the following context to answer the question thoroughly and accurately. "
            f"Break down your answer into clear sections when appropriate. "
            f"Question: {query}\n\n"
            f"Context Information:\n{context_text}\n\n"
            f"Instructions:\n"
            f"1. Answer the question comprehensively using the provided context\n"
            f"2. If specific numbers, dates, or statistics are mentioned in the context, include them\n"
            f"3. If the context contains multiple relevant points, address each one\n"
            f"4. Maintain a professional and informative tone\n"
            f"5. If there are any important caveats or additional details, mention them\n\n"
            f"Detailed Answer:"
        )
        
        try:
            outputs = self.generator(prompt)
            answer = outputs[0]["generated_text"]
            
            # Clean up the answer by removing the prompt if it's included in the output
            if "Detailed Answer:" in answer:
                answer = answer.split("Detailed Answer:")[1].strip()
                
        except Exception as e:
            answer = f"Generation error: {e}"

        return {
            "answer": answer,
            "score": None,
            "thought_process": "Based on program documentation, student community inputs, and official sources",
            "context": context_text,
            "metadata": {
                "sources_used": used_sources,
                "contexts_found": len(top_contexts),
            },
            "source_type": "contextual-qa",
            "faq_entry": {}
        }


# ---------------------------
# Helper Print Function
# ---------------------------

def print_answers(query: str, answers: List[Dict[str, Any]], max_print: int = 1):
    print(f"\nQuestion: {query}")
    print("-" * 80)
    for i, ans in enumerate(answers[:max_print]):
        print(f"\nAnswer #{i+1}:")
        print(f"\n{ans.get('answer', 'No answer found')}")
        
        # Print source information
        print("\nSources Used:")
        sources = ans.get("metadata", {}).get("sources_used", [])
        for src in sources:
            source_type = src.get("source", "unknown")
            url = src.get("url", "")
            question = src.get("question", "")
            print(f"└─ {source_type}")
            if url:
                print(f"   URL: {url}")
            if question and not question.startswith("Web Content #"):
                print(f"   Topic: {question}")
        
        if ans.get('thought_process'):
            print(f"\nReasoning Process:")
            thought = ans['thought_process'].strip()
            print(f"└─ {thought}")
        
        print("-" * 80)


# ---------------------------
# Main Function
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Llama 2 7B Chat Semantic QA Retriever")
    parser.add_argument('--faq', type=str, default=None,
                        help='Path to FAQ JSON file')
    parser.add_argument('--url', type=str, default=None,
                        help='URL to scrape paragraphs from')
    parser.add_argument('--query', nargs='+', required=False,
                        help='Question to ask (can be multiple words)')
    parser.add_argument('--max_print', type=int, default=2,
                        help='Number of answers to print')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top K contexts to retrieve for generation')
    parser.add_argument('--device', type=int, default=0,
                        help='Device index (GPU id). Set to -1 for CPU')
    args = parser.parse_args()

    context_agent = ContextAgent()
    if args.faq:
        context_agent.add_source(JSONContentSource(args.faq))
    if args.url:
        context_agent.add_source(WebContentSource(args.url))

    if not context_agent.get_faq_data():
        print("No content sources provided. Use --faq and/or --url")
        return

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    context_agent.build_semantic_search_index(embedder)

    agent = QAAgent(device=args.device)

    query = ' '.join(args.query) if args.query else input(
        "Enter your question: ").strip()
    if not query:
        print("No query given. Exiting.")
        return

    result = agent.process_query(query, context_agent, top_k=args.top_k)
    print_answers(query, [result], max_print=args.max_print)


if __name__ == "__main__":
    main()
