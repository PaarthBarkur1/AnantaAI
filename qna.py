

import json
import logging
import hashlib
import time
import argparse
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers.pipelines import pipeline
import torch
from webscrapper import WebScraper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# Enhanced Helper Functions
# ---------------------------

def load_sources_config(path: str = "sources.json") -> List[Dict[str, Any]]:
    """Load the sources.json config file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load sources config: {e}")
        return []

def extract_keywords(text: str) -> set:
    """Extract lowercase keywords from a question."""
    return set(word.lower() for word in text.split() if len(word) > 2)

def select_relevant_sources(question: str, sources: List[Dict[str, Any]], top_n: int = 2) -> List[Dict[str, Any]]:
    """Select the most relevant sources for a question based on keyword/category overlap."""
    qwords = extract_keywords(question)
    scored = []
    for src in sources:
        cats = set()
        for c in src.get("category", []):
            cats.update(c.lower().split())
        overlap = len(qwords & cats)
        scored.append((overlap, src))
    scored.sort(reverse=True, key=lambda x: x[0])
    # Always return at least one source, even if no overlap
    return [s[1] for s in scored if s[0] > 0][:top_n] or [scored[0][1]]



# ---------------------------
# Configuration and Data Classes
# ---------------------------

# ---------------------------
# Configuration and Data Classes
# ---------------------------


@dataclass
class QAConfig:
    """Configuration class for QA system"""
    max_length: int = 1024
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    context_window: int = 800
    min_confidence: float = 0.1


@dataclass
class SearchResult:
    """Structured search result"""
    content: Dict[str, Any]
    distance: float
    confidence: float

# ---------------------------
# Content Source Abstract and Implementations
# ---------------------------


class ContentSource(ABC):
    """Abstract base class for content sources"""

    @abstractmethod
    def get_content(self) -> List[Dict[str, Any]]:
        """Get content from the source"""
        pass

    @abstractmethod
    def get_source_type(self) -> str:
        """Get the type of content source"""
        pass

    def validate_content(self, content: List[Dict[str, Any]]) -> bool:
        """Validate content structure"""
        required_fields = {'question', 'answer'}
        return all(
            isinstance(item, dict) and required_fields.issubset(item.keys())
            for item in content
        )


class JSONContentSource(ContentSource):
    """JSON file content source with caching and validation"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self._cache = None
        self._cache_hash = None

    def get_source_type(self) -> str:
        return "json"

    def _get_file_hash(self) -> str:
        """Get hash of file for cache invalidation"""
        if not self.filepath.exists():
            return ""
        return hashlib.md5(self.filepath.read_bytes()).hexdigest()

    def get_content(self) -> List[Dict[str, Any]]:
        """Load content with caching"""
        current_hash = self._get_file_hash()

        # Use cache if available and file hasn't changed
        if self._cache is not None and self._cache_hash == current_hash:
            return self._cache

        try:
            if not self.filepath.exists():
                raise FileNotFoundError(f"File not found: {self.filepath}")

            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("JSON data must be a list of QA pairs")

            # Add metadata to each entry
            for item in data:
                if 'metadata' not in item:
                    item['metadata'] = {}
                item['metadata'].update({
                    'source': self.get_source_type(),
                    'filepath': str(self.filepath)
                })

            if not self.validate_content(data):
                raise ValueError("Invalid content structure in JSON file")

            # Cache the result
            self._cache = data
            self._cache_hash = current_hash

            logger.info(f"Loaded {len(data)} entries from {self.filepath}")
            return data

        except Exception as e:
            logger.error(f"Failed to load JSON from {self.filepath}: {e}")
            raise RuntimeError(
                f"Failed to load JSON data from {self.filepath}: {e}")


class WebContentSource(ContentSource):
    """Enhanced web scraping content source with comprehensive content extraction"""

    def __init__(self, url: str, max_retries: int = 3):
        self.url = url
        self.max_retries = max_retries
        self.scraper = WebScraper()
        self._cache = None

    def get_source_type(self) -> str:
        return "web"

    def get_content(self) -> List[Dict[str, Any]]:
        """Scrape content with enhanced capabilities and better structure"""
        if self._cache is not None:
            return self._cache

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Scraping content from {self.url} (attempt {attempt + 1})")
                content = []

                # Get paragraphs with main content
                paragraphs = self.scraper.scrape_paragraphs(self.url)
                for i, para in enumerate(paragraphs):
                    if para.strip() and len(para.strip()) > 50:  # Filter meaningful paragraphs
                        content.append({
                            "question": f"Main Content Section {i + 1}",
                            "answer": para.strip(),
                            "metadata": {
                                "source": self.get_source_type(),
                                "url": self.url,
                                "type": "paragraph",
                                "section_id": i + 1
                            }
                        })

                # Get headlines for structure
                headlines = self.scraper.scrape_headlines(self.url)
                for i, headline in enumerate(headlines):
                    content.append({
                        "question": f"Section Heading {i + 1}",
                        "answer": headline["text"],
                        "metadata": {
                            "source": self.get_source_type(),
                            "url": self.url,
                            "type": "headline",
                            "level": headline["level"]
                        }
                    })

                # Try to get structured table data
                tables = self.scraper.scrape_table_data(self.url)
                for i, table in enumerate(tables):
                    if table:  # Only process non-empty tables
                        table_content = "\n".join([" | ".join(row) for row in table])
                        content.append({
                            "question": f"Table Content {i + 1}",
                            "answer": table_content,
                            "metadata": {
                                "source": self.get_source_type(),
                                "url": self.url,
                                "type": "table",
                                "table_id": i + 1
                            }
                        })

                # Get WordPress recent posts if available
                try:
                    recent_posts = self.scraper.scrape_wordpress_recent_posts(self.url)
                    for i, post in enumerate(recent_posts):
                        content.append({
                            "question": f"Recent Post {i + 1}",
                            "answer": f"Title: {post['title']}\nLink: {post['link']}",
                            "metadata": {
                                "source": self.get_source_type(),
                                "url": self.url,
                                "type": "wordpress_post",
                                "post_id": i + 1
                            }
                        })
                except Exception as wp_error:
                    logger.debug(f"Not a WordPress site or no recent posts: {wp_error}")

                # Try specific content areas if no content found yet
                if not content:
                    specific_content = self.scraper.scrape_specific_element(self.url, "div.entry-content, article, .content-area")
                    for i, text in enumerate(specific_content):
                        if text.strip() and len(text.strip()) > 50:
                            content.append({
                                "question": f"Specific Content Section {i + 1}",
                                "answer": text.strip(),
                                "metadata": {
                                    "source": self.get_source_type(),
                                    "url": self.url,
                                    "type": "specific_content",
                                    "section_id": i + 1
                                }
                            })

                if not content:
                    raise ValueError("No content found at URL")

                logger.info(f"Successfully scraped {len(content)} items from {self.url}")
                self._cache = content
                return content

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Failed to scrape content from {self.url} after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        return []

# ---------------------------
# Enhanced Context Agent with Advanced Search
# ---------------------------


class ContextAgent:
    """Enhanced context agent with semantic search and intelligent ranking"""

    def __init__(self, config: Optional[QAConfig] = None):
        self.config = config or QAConfig()
        self.sources: List[ContentSource] = []
        self.faq_data: List[Dict[str, Any]] = []

        # Search components
        self.embedder: Optional[SentenceTransformer] = None
        self.doc_embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []

        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0
        }

    def add_source(self, source: ContentSource) -> None:
        """Add a content source and load its data"""
        logger.info(f"Adding {source.get_source_type()} source")
        self.sources.append(source)
        new_data = source.get_content()
        self.faq_data.extend(new_data)
        logger.info(f"Total entries: {len(self.faq_data)}")

    def get_faq_data(self) -> List[Dict[str, Any]]:
        return self.faq_data

    def get_source_confidence(self, source_type: str) -> float:
        """Get confidence score based on source type"""
        confidence_scores = {
            "json": 0.95,
            "web": 0.75,
        }
        return confidence_scores.get(source_type, 0.5)

    def _create_document_text(self, entry: Dict[str, Any]) -> str:
        """Create searchable document text from entry"""
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        source = entry.get("metadata", {}).get("source", "unknown")

        # Create more searchable text
        if question.startswith("Web Content"):
            return f"Content: {answer}\nSource: {source}"
        else:
            return f"Question: {question}\nAnswer: {answer}\nSource: {source}"

    def is_ready(self) -> bool:
        """Check if the context agent is ready for searching"""
        return (
            self.index is not None and
            self.embedder is not None and
            len(self.faq_data) > 0 and
            len(self.documents) > 0
        )

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of the context agent"""
        return {
            "has_data": len(self.faq_data) > 0,
            "has_embedder": self.embedder is not None,
            "has_index": self.index is not None,
            "document_count": len(self.documents),
            "entry_count": len(self.faq_data),
            "ready": self.is_ready()
        }

    def build_semantic_search_index(self, embedder: SentenceTransformer) -> None:
        """Build semantic search index with optimization"""
        logger.info("Building semantic search index...")
        start_time = time.time()

        self.embedder = embedder
        self.documents = []

        # Create document texts
        for entry in self.faq_data:
            doc_text = self._create_document_text(entry)
            self.documents.append(doc_text)

        if not self.documents:
            raise ValueError("No documents to index for semantic search.")

        # Generate embeddings in batches for efficiency
        batch_size = 32
        embeddings = []

        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            batch_embeddings = embedder.encode(
                batch, convert_to_tensor=False, show_progress_bar=False)
            embeddings.append(batch_embeddings)

        self.doc_embeddings = np.vstack(embeddings)

        # Create FAISS index with better performance
        dim = self.doc_embeddings.shape[1]
        if len(self.documents) > 1000:
            # Use IVF index for large datasets
            nlist = min(100, len(self.documents) // 10)
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)
            train_data = self.doc_embeddings.astype(np.float32)
            assert train_data.ndim == 2, f"train_data must be 2D, got shape {train_data.shape}"
            # FAISS expects a 2D np.float32 array
            self.index.train(train_data.reshape(-1, dim))
        else:
            # Use flat index for smaller datasets
            self.index = faiss.IndexFlatL2(dim)

        add_data = self.doc_embeddings.astype(np.float32)
        assert add_data.ndim == 2, f"add_data must be 2D, got shape {add_data.shape}"
        # FAISS expects a 2D np.float32 array
        self.index.add(add_data)

        build_time = time.time() - start_time
        logger.info(
            f"Built semantic index with {len(self.documents)} documents in {build_time:.2f}s")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Enhanced search with confidence scoring"""
        if self.index is None or self.embedder is None:
            raise RuntimeError(
                "Semantic index not built. Call build_semantic_search_index() first.")

        start_time = time.time()

        # Generate query embedding
        query_emb = self.embedder.encode([query], convert_to_tensor=False)
        query_emb = np.array(query_emb).astype(np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # Search with more candidates for reranking
        search_k = min(top_k * 2, len(self.faq_data))
        # query_emb: shape (1, dim), search_k: int
        # FAISS search
        distances, indices = self.index.search(query_emb, int(search_k))

        # Create search results with confidence scores
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.faq_data):
                entry = self.faq_data[idx]
                source_type = entry.get(
                    "metadata", {}).get("source", "unknown")

                # Calculate confidence (convert L2 distance to similarity)
                similarity = 1 / (1 + distance)
                source_confidence = self.get_source_confidence(source_type)
                combined_confidence = similarity * source_confidence

                results.append(SearchResult(
                    content=entry,
                    distance=distance,
                    confidence=combined_confidence
                ))

        # Sort by confidence and filter
        results.sort(key=lambda x: x.confidence, reverse=True)
        filtered_results = [
            r for r in results if r.confidence >= self.config.min_confidence]

        # Update stats
        search_time = time.time() - start_time
        self.search_stats['total_searches'] += 1
        self.search_stats['avg_search_time'] = (
            (self.search_stats['avg_search_time'] *
             (self.search_stats['total_searches'] - 1) + search_time)
            / self.search_stats['total_searches']
        )

        return filtered_results[:top_k]

# ---------------------------
# Enhanced QA Agent with Qwen2.5 Model
# ---------------------------


class QAAgent:
    """Enhanced QA Agent with Qwen2.5 model and intelligent section extraction"""

    def __init__(self, device: int = 0, config: Optional[QAConfig] = None):
        self.config = config or QAConfig()
        self.device = device

        # Initialize tokenizer for Qwen2.5
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Successfully loaded Qwen2.5 tokenizer")
        except Exception as e:
            logger.error(f"Failed to load Qwen tokenizer: {e}")
            # Fallback to context extraction only
            self.use_ai_generation = False
            logger.info("Falling back to context extraction mode")
            return

        # Check device availability and set device properly
        self.device = device if (torch.cuda.is_available() and device >= 0) else -1
        device_name = torch.cuda.get_device_name(self.device) if self.device >= 0 else "CPU"
        logger.info(f"Using device: {device_name}")

        # Set device map for pipeline initialization
        device_map = "auto" if self.device >= 0 else None
        torch_device = f"cuda:{self.device}" if self.device >= 0 else "cpu"
        logger.info(f"Using device map: {device_map}, torch device: {torch_device}")

        # Initialize Qwen2.5 generation pipeline
        try:
            self.generator = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                tokenizer=self.tokenizer,
                device_map=device_map,  # Use device_map for better GPU utilization
                torch_dtype=torch.float16 if self.device >= 0 else None,  # Use FP16 on GPU
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                trust_remote_code=True  # Required for Qwen models
            )
            self.use_ai_generation = True
            logger.info(
                "Successfully initialized Qwen2.5 model for text generation")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen2.5 model: {e}")
            self.use_ai_generation = False
            logger.info("Falling back to context extraction mode")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=True))
        except:
            # Fallback: rough estimation
            return int(len(text.split()) * 1.3)

    def _create_qwen_prompt(self, query: str, context: str) -> str:
        """Create a proper prompt for Qwen2.5 model"""
        return f"""<|im_start|>system
You are a helpful assistant answering questions about the IISc M.Mgt program. Provide clear, accurate answers based on the given context.
<|im_end|>
<|im_start|>user
Context: {context[:600]}

Question: {query}

Please provide a clear and concise answer based on the context above.
<|im_end|>
<|im_start|>assistant
"""

    def _clean_qwen_output(self, text: str) -> str:
        """Clean Qwen model output"""
        # Remove special tokens
        text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
        text = text.replace("<|endoftext|>", "")

        # Clean up any remaining artifacts
        text = text.strip()

        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'

        return text

    def _extract_relevant_sentences(self, query: str, text: str, max_sentences: int = 4) -> str:
        """Extract the most relevant sentences from a large text block"""
        if not text:
            return ""

        query_lower = query.lower()
        query_words = set(word.lower()
                          for word in query.split() if len(word) > 2)

        # Split text into sentences
        sentences = []
        for sent in text.split('.'):
            sent = sent.strip()
            if len(sent) > 20:  # Minimum meaningful sentence length
                sentences.append(sent)

        # Score each sentence based on relevance
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())

            # Calculate different types of matches
            exact_matches = sum(
                1 for word in query_words if word in sentence_lower)
            word_overlap = len(query_words.intersection(sentence_words))

            # Bonus for sentences that contain key terms
            key_terms = {
                'eligibility': ['degree', 'bachelor', 'engineering', 'marks', 'cgpa'],
                'placement': ['ctc', 'salary', 'companies', 'placed', 'offers'],
                'interview': ['interview', 'questions', 'math', 'probability', 'discussion'],
                'student life': ['campus', 'classes', 'facilities', 'accommodation'],
                'curriculum': ['courses', 'credits', 'subjects', 'electives'],
                'admission': ['percentile', 'cutoff', 'cat', 'gate']
            }

            bonus_score = 0
            for category, terms in key_terms.items():
                if any(term in query_lower for term in [category]):
                    bonus_score += sum(1 for term in terms if term in sentence_lower) * 0.5

            total_score = exact_matches * 2 + word_overlap + bonus_score
            sentence_scores.append((sentence, total_score, i))

        # Sort by score and position (prefer earlier sentences for ties)
        sentence_scores.sort(key=lambda x: (-x[1], x[2]))

        # Select top sentences
        selected_sentences = []
        for sentence, score, _ in sentence_scores[:max_sentences]:
            if score > 0:  # Only include sentences with some relevance
                selected_sentences.append(sentence)

        if selected_sentences:
            result = '. '.join(selected_sentences)
            if not result.endswith('.'):
                result += '.'
            return result

        return ""

    def _generate_answer_with_qwen(self, query: str, context: str) -> str:
        """Generate answer using Qwen2.5 model"""
        if not self.use_ai_generation:
            return ""

        try:
            # Create proper Qwen prompt
            prompt = self._create_qwen_prompt(query, context)

            # Check token count
            if self._count_tokens(prompt) > self.config.max_length - self.config.max_new_tokens:
                # Truncate context if prompt is too long
                context = context[:400] + "..."
                prompt = self._create_qwen_prompt(query, context)

            # Generate with Qwen
            outputs = self.generator(
                prompt,
                max_new_tokens=min(150, self.config.max_new_tokens),
                temperature=0.3,  # Lower temperature for more focused answers
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            if outputs and len(outputs) > 0:
                generated_text = outputs[0].get("generated_text", "")
                cleaned_answer = self._clean_qwen_output(generated_text)

                # Validate the output
                if len(cleaned_answer) > 10 and cleaned_answer.count(' ') > 3:
                    return cleaned_answer

        except Exception as e:
            logger.warning(f"Qwen generation failed: {e}")

        return ""

    def _extract_direct_answer(self, query: str, search_results: List) -> str:
        """Extract focused answer from search results with AI generation fallback"""
        if not search_results:
            return "I couldn't find relevant information to answer your question."

        # Get the best contexts
        contexts = []
        for result in search_results[:3]:  # Top 3 results
            full_answer = result.content.get('answer', '')
            if full_answer:
                contexts.append(full_answer)

        combined_context = " ".join(contexts)

        # Try AI generation first if available
        if self.use_ai_generation and combined_context:
            ai_answer = self._generate_answer_with_qwen(
                query, combined_context)
            if ai_answer:
                return ai_answer

        # Fallback to sentence extraction
        best_result = search_results[0]
        full_answer = best_result.content.get('answer', '')

        if not full_answer:
            return "I found some information but couldn't extract a specific answer."

        # Extract relevant section
        relevant_section = self._extract_relevant_sentences(
            query, full_answer, max_sentences=3)

        if relevant_section:
            return relevant_section
        else:
            # Final fallback: return first few sentences
            sentences = full_answer.split('.')[:3]
            return '. '.join(sent.strip() for sent in sentences if sent.strip()) + '.'

    def process_query(self, query: str, context_agent: ContextAgent, top_k: int = 5) -> Dict[str, Any]:
        """Process query with Qwen2.5 generation and focused section extraction"""
        start_time = time.time()

        try:
            # Search for relevant contexts
            search_results = context_agent.search(query, top_k=top_k)

            if not search_results:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic related to the IISc M.Mgt program.",
                    "confidence": 0.0,
                    "context": "",
                    "metadata": {"sources_used": [], "contexts_found": 0, "method": "no_results", "model": "qwen2.5"},
                    "source_type": "no-match",
                    "processing_time": time.time() - start_time
                }

            # Extract focused answer (with AI generation if available)
            answer = self._extract_direct_answer(query, search_results)

            # Prepare metadata with focused context
            used_sources = []
            focused_contexts = []

            for result in search_results:
                content = result.content
                source_type = content.get(
                    "metadata", {}).get("source", "unknown")

                # Extract relevant section for context display
                full_text = content.get('answer', '')
                relevant_section = self._extract_relevant_sentences(
                    query, full_text, max_sentences=2)

                if relevant_section:
                    focused_contexts.append(relevant_section)

                used_sources.append({
                    "source": source_type,
                    "url": content.get("metadata", {}).get("url", ""),
                    "question": content.get("question", ""),
                    "confidence": result.confidence
                })

            combined_context = "\n\n".join(focused_contexts[:2])
            avg_confidence = sum(
                r.confidence for r in search_results) / len(search_results)
            processing_time = time.time() - start_time

            generation_method = "qwen_ai_generation" if self.use_ai_generation else "section_extraction"

            return {
                "answer": answer,
                "confidence": avg_confidence,
                "context": combined_context,
                "metadata": {
                    "sources_used": used_sources,
                    "contexts_found": len(search_results),
                    "method": generation_method,
                    "model": "qwen2.5-0.5b-instruct",
                    "processing_time": processing_time
                },
                "source_type": "contextual-qa",
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again or rephrase your question.",
                "confidence": 0.0,
                "context": "",
                "metadata": {"error": str(e), "sources_used": [], "contexts_found": 0, "method": "error", "model": "qwen2.5"},
                "source_type": "error",
                "processing_time": time.time() - start_time
            }

# ---------------------------
# Enhanced Helper Functions
# ---------------------------


def print_answers(query: str, answers: List[Dict[str, Any]], max_print: int = 1, verbose: bool = False):
    """Enhanced answer printing with better formatting"""
    print(f"\n{'='*80}")
    print(f"Question: {query}")
    print(f"{'='*80}")

    for i, ans in enumerate(answers[:max_print]):
        print(f"\n📝 Answer #{i+1}:")
        print("-" * 60)

        answer_text = ans.get('answer', 'No answer found')
        print(f"\n{answer_text}")

        # Show confidence if available
        confidence = ans.get('confidence', 0)
        if confidence > 0:
            print(f"\n🎯 Confidence: {confidence:.1%}")

        # Show processing time
        proc_time = ans.get('processing_time', 0)
        if proc_time > 0:
            print(f"⏱️  Processing Time: {proc_time:.2f}s")

        # Show model used
        metadata = ans.get("metadata", {})
        model_used = metadata.get("model", "unknown")
        method = metadata.get("method", "unknown")
        print(f"🤖 Model: {model_used} ({method})")

        # Show sources
        sources = metadata.get("sources_used", [])
        if sources:
            print(f"\n📚 Sources Used ({len(sources)}):")
            for j, src in enumerate(sources, 1):
                source_type = src.get("source", "unknown")
                confidence = src.get("confidence", 0)
                truncated = src.get("truncated", False)

                print(f"  {j}. {source_type.upper()}")
                if confidence > 0:
                    print(f"     Confidence: {confidence:.1%}")
                if truncated:
                    print(f"     ⚠️  Content was truncated to fit")

                url = src.get("url", "")
                if url:
                    print(f"     URL: {url}")

                question = src.get("question", "")
                if question and not question.startswith("Web Content"):
                    print(f"     Topic: {question}")

        # Show technical details if verbose
        if verbose:
            print(f"\n🔧 Technical Details:")
            contexts_found = metadata.get("contexts_found", 0)
            print(f"  Contexts Found: {contexts_found}")
            print(f"  Generation Method: {method}")
            print(f"  Model: {model_used}")

            if ans.get('source_type'):
                print(f"  Source Type: {ans['source_type']}")

        print("-" * 60)


def create_config_from_args(args) -> QAConfig:
    """Create configuration from command line arguments"""
    return QAConfig(
        max_length=getattr(args, 'max_length', 1024),
        max_new_tokens=getattr(args, 'max_new_tokens', 256),
        temperature=getattr(args, 'temperature', 0.7),
        context_window=getattr(args, 'context_window', 800),
        min_confidence=getattr(args, 'min_confidence', 0.1)
    )

# ---------------------------
# Enhanced Main Function
# ---------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Semantic QA System with Qwen2.5 Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qna.py --faq context.json --query "What are the eligibility criteria?"
  python qna.py --url "https://example.com" --query "Tell me about the program"
  python qna.py --faq context.json --url "https://example.com" --interactive
        """
    )

    # Data sources
    parser.add_argument('--faq', type=str, help='Path to FAQ JSON file')
    parser.add_argument('--url', type=str, help='URL to scrape content from')

    # Query options
    parser.add_argument('--query', nargs='+', help='Question to ask')
    parser.add_argument('--interactive', action='store_true',
                        help='Start interactive mode')

    # Model parameters
    parser.add_argument('--device', type=int, default=0,
                        help='Device index (-1 for CPU)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top K contexts to retrieve')
    parser.add_argument('--max_length', type=int,
                        default=1024, help='Maximum total tokens')
    parser.add_argument('--max_new_tokens', type=int,
                        default=256, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float,
                        default=0.7, help='Generation temperature')
    parser.add_argument('--context_window', type=int,
                        default=800, help='Maximum context tokens')
    parser.add_argument('--min_confidence', type=float,
                        default=0.1, help='Minimum confidence threshold')

    # Output options
    parser.add_argument('--max_print', type=int, default=1,
                        help='Number of answers to print')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')

    args = parser.parse_args()


    # If neither --faq nor --url is provided, load all sources from sources.json
    if not args.faq and not args.url:
        logger.info("No --faq or --url provided. Loading all sources from sources.json for agentic mode.")
        load_all_sources = True
    else:
        load_all_sources = False

    if not args.query and not args.interactive:
        parser.error("Either --query or --interactive must be provided")

    try:
        # Create configuration
        config = create_config_from_args(args)

        # Initialize context agent
        logger.info("Initializing context agent...")
        context_agent = ContextAgent(config)

        # Add sources
        if load_all_sources:
            try:
                with open("sources.json", "r", encoding="utf-8") as f:
                    sources = json.load(f)
                    for src in sources:
                        try:
                            web_source = WebContentSource(src["url"])
                            context_agent.add_source(web_source)
                        except Exception as e:
                            logger.warning(f"Failed to add source {src.get('name', 'Unknown')}: {e}")
            except Exception as e:
                logger.error(f"Failed to load sources.json: {e}")
        else:
            if args.faq:
                context_agent.add_source(JSONContentSource(args.faq))
            if args.url:
                context_agent.add_source(WebContentSource(args.url))

        if not context_agent.get_faq_data():
            logger.error("No content loaded from sources")
            return

        # Build semantic index
        logger.info("Loading embedding model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        context_agent.build_semantic_search_index(embedder)

        # Initialize QA agent with Qwen2.5
        logger.info("Initializing QA agent with Qwen2.5 model...")
        qa_agent = QAAgent(device=args.device, config=config)

        # Process queries
        if args.interactive:
            print("\n🤖 Interactive QA System with Qwen2.5 (type 'quit' to exit)")
            print("=" * 50)

            while True:
                try:
                    query = input("\n❓ Your question: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break

                    if query:
                        result = qa_agent.process_query(
                            query, context_agent, top_k=args.top_k)
                        print_answers(
                            query, [result], max_print=args.max_print, verbose=args.verbose)

                except KeyboardInterrupt:
                    print("\n\nGoodbye! 👋")
                    break
                except Exception as e:
                    logger.error(f"Error in interactive mode: {e}")
                    print(f"❌ Error: {e}")

        else:
            query = ' '.join(args.query)
            logger.info(f"Processing query: {query}")
            result = qa_agent.process_query(
                query, context_agent, top_k=args.top_k)
            print_answers(
                query, [result], max_print=args.max_print, verbose=args.verbose)

        # Show performance stats
        if args.verbose:
            stats = context_agent.search_stats
            print(f"\n📊 Performance Stats:")
            print(f"Total searches: {stats['total_searches']}")
            print(f"Average search time: {stats['avg_search_time']:.3f}s")

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
