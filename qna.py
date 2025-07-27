import json
import argparse
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Any, Optional

def load_faq_data(filepath: str) -> List[Dict[str, Any]]:
    """Load FAQ data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAQ data: {e}")
    if not faq_data or not isinstance(faq_data, list):
        raise ValueError("FAQ data is empty or not a list.")
    return faq_data

def build_documents(faq_data: List[Dict[str, Any]]) -> List[str]:
    """Extract answers (or questions+answers) for retrieval."""
    docs = []
    for entry in faq_data:
        answer = entry.get('answer', '')
        question = entry.get('question', '')
        if answer and question:
            docs.append(f"Q: {question}\nA: {answer}")
        elif answer:
            docs.append(answer)
        elif question:
            docs.append(question)
        else:
            docs.append('')
    return docs

def build_index(documents: List[str], embedder) -> faiss.IndexFlatL2:
    """Build FAISS index from document embeddings."""
    doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    return index, doc_embeddings

def answer_query(query: str, faq_data: List[Dict[str, Any]], documents: List[str],
                 embedder, index, qa_pipeline, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve top_k answers to a query from the FAQ using semantic search and QA model.
    Returns a list of answer dicts sorted by QA score.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string.")

    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)

    candidates = []
    for idx in indices[0]:
        context = documents[idx]
        result = qa_pipeline(question=query, context=context)
        candidates.append({
            'answer': result.get('answer', ''),
            'score': result.get('score', 0.0),
            'context': context,
            'metadata': faq_data[idx].get('metadata', {}),
            'faq_entry': faq_data[idx]
        })
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates

def print_answers(query: str, answers: List[Dict[str, Any]], max_print: int = 1):
    """Pretty-print the top answers for a query."""
    print(f"\nQuestion: {query}")
    for i, ans in enumerate(answers[:max_print]):
        print(f"\nTop Answer #{i+1}:")
        print(f"Answer: {ans['answer']}")
        print(f"Score: {ans['score']:.4f}")
        print(f"Source Info: {ans['metadata']}")
        faq_entry = ans.get('faq_entry', {})
        if 'question' in faq_entry:
            print(f"FAQ Question: {faq_entry['question']}")
        # Uncomment to print context
        # print(f"Context: {ans['context']}")

def main():
    parser = argparse.ArgumentParser(description="FAQ Semantic QA Retriever")
    parser.add_argument('--faq', type=str, default='.json', help='Path to FAQ JSON file')
    parser.add_argument('--query', nargs='+', help='Question to ask (can be multiple words)')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top answers to retrieve')
    parser.add_argument('--max_print', type=int, default=2, help='Number of answers to print')
    args = parser.parse_args()

    faq_data = load_faq_data(args.faq)
    documents = build_documents(faq_data)

    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        raise RuntimeError(f"Failed to load SentenceTransformer: {e}")

    index, _ = build_index(documents, embedder)

    model_name = "deepset/roberta-base-squad2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    except Exception as e:
        raise RuntimeError(f"Failed to load QA model: {e}")

    # Join query words if provided
    query = ' '.join(args.query) if args.query else input("Enter your question: ")
    answers = answer_query(query, faq_data, documents, embedder, index, qa_pipeline, top_k=args.top_k)
    print_answers(query, answers, max_print=args.max_print)

if __name__ == "__main__":
    main()
