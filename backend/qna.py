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
from .webscrapper import WebScraper

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
    """Enhanced configuration class for QA system with adaptive settings"""
    max_length: int = 1024
    max_new_tokens: int = 256
    temperature: float = 0.3  # Balanced - not too rigid, not too creative
    top_p: float = 0.8  # Allow some variety in responses
    do_sample: bool = True
    context_window: int = 800
    min_confidence: float = 0.05  # Lower threshold for better recall
    use_ai_generation: bool = True  # Set to False to disable AI model loading

    # Enhanced configuration options
    embedding_model: str = "auto"  # Auto-select based on resources
    max_search_results: int = 8  # Increased for better context selection
    # intelligent, simple, comprehensive
    context_fusion_strategy: str = "intelligent"
    answer_quality_threshold: float = 0.15  # Minimum quality for AI answers
    enable_query_expansion: bool = True  # Enable enhanced query preprocessing
    enable_context_reranking: bool = True  # Enable intelligent context reranking

    # Performance tuning
    sentence_extraction_max: int = 4  # Max sentences for extraction
    # Threshold for avoiding duplicate contexts
    context_overlap_threshold: float = 0.7
    numerical_info_boost: float = 1.5  # Boost for numerical information relevance


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

                # --- ENHANCEMENT: Use custom scrapers first based on URL keywords ---
                if "landing_page" in self.url:
                    data = self.scraper.scrape_landing_page_content(self.url)
                    # Convert structured data from the custom scraper into the content format
                    if data:
                        content.append({
                            "question": "Landing Page Hero Section",
                            "answer": f"{data.get('hero_title', '')}\n{data.get('hero_subtitle', '')}",
                            "metadata": {"source": self.get_source_type(), "url": self.url, "type": "hero_section"}
                        })
                        if data.get('statistics'):
                            stats_text = "\n".join([f"{s['value']} - {s['label']}" for s in data['statistics']])
                            content.append({
                                "question": "Landing Page Statistics",
                                "answer": stats_text,
                                "metadata": {"source": self.get_source_type(), "url": self.url, "type": "statistics"}
                            })
                        if data.get('testimonials'):
                            testimonials_text = "\n\n".join([f"\"{t['quote']}\" - {t['author']}" for t in data['testimonials']])
                            content.append({
                                "question": "Customer Testimonials",
                                "answer": testimonials_text,
                                "metadata": {"source": self.get_source_type(), "url": self.url, "type": "testimonials"}
                            })

                elif "recruiter-insights" in self.url:
                    data = self.scraper.scrape_recruiter_insights_page(self.url)
                    if data:
                        content.append({
                            "question": "Recruiter Insights Page Title",
                            "answer": data.get('page_title'),
                            "metadata": {"source": self.get_source_type(), "url": self.url, "type": "page_title"}
                        })
                        if data.get('insights'):
                            insights_text = "\n\n".join([f"Title: {i['title']}\nDescription: {i['description']}" for i in data['insights']])
                            content.append({
                                "question": "Recruiter Insights Sections",
                                "answer": insights_text,
                                "metadata": {"source": self.get_source_type(), "url": self.url, "type": "insights"}
                            })

                elif "contact" in self.url:
                    data = self.scraper.scrape_contact_page_content(self.url)
                    if data:
                        content.append({
                            "question": "Contact Page Information",
                            "answer": f"Heading: {data.get('heading')}\nForm Fields: {', '.join(data.get('form_fields', []))}",
                            "metadata": {"source": self.get_source_type(), "url": self.url, "type": "contact_info"}
                        })
                        if data.get('contact_info'):
                            contact_details = "\n".join([f"{k}: {v}" for k, v in data['contact_info'].items()])
                            content.append({
                                "question": "Contact Details",
                                "answer": contact_details,
                                "metadata": {"source": self.get_source_type(), "url": self.url, "type": "contact_details"}
                            })

                # --- FALLBACK: If no custom scraper matched, use the generic methods ---
                if not content:
                    # Get paragraphs with main content
                    paragraphs = self.scraper.scrape_paragraphs(self.url)
                    for i, para in enumerate(paragraphs):
                        if para.strip() and len(para.strip()) > 50:
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
                        if table:
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
                    raise RuntimeError(f"Failed to scrape content from {self.url} after {self.max_retries} attempts: {e}")
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
            "json": 1.0,   # Maximum confidence for curated FAQ data
            "web": 0.75,   # Lower confidence for web content
            "official": 0.85,  # High confidence for official sources
            "faq": 1.0,    # Maximum confidence for FAQ content
            "unknown": 0.5  # Lower confidence for unknown sources
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

    def _preprocess_query(self, query: str) -> str:
        """Enhanced query preprocessing with better understanding and expansion"""
        # Convert to lowercase for processing
        query_lower = query.lower().strip()

        # Normalize common variations and typos
        query_normalized = self._normalize_query(query_lower)

        # Detect basic query type for expansion
        query_type = self._detect_query_type(query_normalized)

        # Create simple intent structure for compatibility
        query_intent = {
            'type': query_type,
            'specificity': 'general',
            'information_type': 'factual',
            'urgency': 'normal'
        }

        # Expand with domain-specific synonyms and related terms
        expanded_query = self._expand_query_with_synonyms(
            query_normalized, query_intent)

        # Add contextual terms based on query type
        contextual_query = self._add_contextual_terms(
            expanded_query, query_intent)

        return contextual_query

    def _normalize_query(self, query: str) -> str:
        """Normalize query by fixing common variations and typos"""
        normalizations = {
            # Common typos and variations
            'mmgt': 'm.mgt',
            'mgt': 'm.mgt',
            'managment': 'management',
            'eligibilty': 'eligibility',
            'critria': 'criteria',
            'placements': 'placement',
            'curriculam': 'curriculum',
            'admision': 'admission',
            'requirments': 'requirements',
            'internships': 'internship',

            # Standardize terms
            'iisc bangalore': 'iisc',
            'indian institute of science': 'iisc',
            'master of management': 'm.mgt',
            'masters in management': 'm.mgt',
            'post graduation': 'postgraduate',
            'pg': 'postgraduate'
        }

        normalized = query
        for wrong, correct in normalizations.items():
            normalized = normalized.replace(wrong, correct)

        return normalized

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query to provide better instructions"""
        query_lower = query.lower()

        # Define query type patterns
        patterns = {
            'eligibility': ['eligibility', 'criteria', 'requirement', 'qualify', 'eligible'],
            'admission': ['admission', 'apply', 'application', 'deadline', 'cutoff', 'cat', 'gate'],
            'curriculum': ['course', 'curriculum', 'subject', 'credit', 'semester', 'elective'],
            'placement': ['placement', 'job', 'salary', 'ctc', 'company', 'recruit'],
            'fees': ['fee', 'cost', 'tuition', 'expense', 'scholarship', 'financial'],
            'campus': ['campus', 'hostel', 'accommodation', 'facility', 'life'],
            'interview': ['interview', 'selection', 'process', 'group discussion'],
            'duration': ['duration', 'year', 'time', 'long', 'period']
        }

        for query_type, keywords in patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type

        return 'general'

    def _expand_query_with_synonyms(self, query: str, intent: dict) -> str:
        """Expand query with comprehensive synonyms and related terms"""
        # Enhanced synonym dictionary with more comprehensive mappings
        synonyms = {
            # Admission related
            "admission": "eligibility application apply entrance selection",
            "eligibility": "criteria requirements qualification minimum",
            "requirements": "criteria eligibility prerequisites conditions",
            "apply": "application admission registration enrollment",
            "cutoff": "percentile score marks minimum threshold",
            "deadline": "last date application window timeline",

            # Academic related
            "curriculum": "courses subjects syllabus academic program structure",
            "courses": "subjects curriculum academic modules classes",
            "credits": "credit hours academic units course load",
            "semester": "term academic period duration",
            "electives": "optional courses choice subjects specialization",
            "core": "mandatory required compulsory essential",

            # Career related
            "placement": "jobs career recruitment employment opportunities",
            "salary": "ctc package compensation pay remuneration",
            "companies": "recruiters employers organizations firms",
            "internship": "summer placement industry exposure training",

            # Financial
            "fees": "cost tuition expenses charges financial",
            "scholarship": "financial aid funding assistance support",

            # Campus life
            "campus": "college institute university facilities infrastructure",
            "hostel": "accommodation residence housing dormitory",
            "facilities": "amenities infrastructure resources services",

            # Time related
            "duration": "length time period years semesters",
            "schedule": "timetable timing classes academic calendar"
        }

        expanded_query = query

        # Add synonyms based on detected terms
        for term, expansion in synonyms.items():
            if term in query:
                # Weight expansion based on intent specificity
                if intent['specificity'] == 'specific':
                    # For specific queries, add fewer but more relevant terms
                    expansion_words = expansion.split()[:3]
                else:
                    # For general queries, add more comprehensive terms
                    expansion_words = expansion.split()

                expanded_query += " " + " ".join(expansion_words)

        return expanded_query

    def _add_contextual_terms(self, query: str, intent: dict) -> str:
        """Add contextual terms based on query type and intent"""
        contextual_terms = {
            'eligibility': ['bachelor degree engineering technology 60% marks cgpa'],
            'admission': ['cat gate percentile application process selection'],
            'curriculum': ['management analytics operations marketing finance'],
            'placement': ['data science analytics consulting technology companies'],
            'fees': ['tuition semester payment installment financial'],
            'campus': ['iisc bangalore facilities hostel single room'],
            'interview': ['group discussion written test math probability'],
            'duration': ['two year full time postgraduate program']
        }

        query_type = intent['type']
        if query_type in contextual_terms:
            # Add contextual terms based on intent urgency
            terms = contextual_terms[query_type]
            if intent['urgency'] == 'urgent':
                # For urgent queries, add specific actionable terms
                contextual_query = query + " " + " ".join(terms[:2])
            else:
                # For normal queries, add comprehensive context
                contextual_query = query + " " + " ".join(terms)
        else:
            contextual_query = query

        return contextual_query

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Enhanced search with confidence scoring and query preprocessing"""
        if self.index is None or self.embedder is None:
            raise RuntimeError(
                "Semantic index not built. Call build_semantic_search_index() first.")

        start_time = time.time()

        # Preprocess query for better matching
        expanded_query = self._preprocess_query(query)

        # Generate query embedding
        query_emb = self.embedder.encode(
            [expanded_query], convert_to_tensor=False)
        query_emb = np.array(query_emb).astype(np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # Search with more candidates for reranking (enhanced)
        search_k = min(self.config.max_search_results, len(self.faq_data))
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

                # Calculate confidence with improved scoring
                similarity = max(0.0, 1.0 - (distance / 1.5))
                source_confidence = self.get_source_confidence(source_type)

                # Enhanced boost logic based on source type and distance
                if source_type == "json":
                    # Higher boost for curated FAQ data
                    similarity = min(1.0, similarity * 1.3)
                elif distance < 0.25:  # Very close matches
                    similarity = min(1.0, similarity * 1.25)
                elif distance < 0.4:  # Good matches
                    similarity = min(1.0, similarity * 1.1)

                # Additional boost for domain-specific content
                if self._is_domain_specific_content(entry, query):
                    similarity = min(1.0, similarity * 1.15)

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

        # Apply diversity filtering to avoid similar results
        diverse_results = self._apply_diversity_filtering(filtered_results, query)

        # Update stats
        search_time = time.time() - start_time
        self.search_stats['total_searches'] += 1
        self.search_stats['avg_search_time'] = (
            (self.search_stats['avg_search_time'] *
             (self.search_stats['total_searches'] - 1) + search_time)
            / self.search_stats['total_searches']
        )

        return diverse_results[:top_k]

    def _is_domain_specific_content(self, entry: dict, query: str) -> bool:
        """Check if content is specifically relevant to IISc M.Mgt domain"""
        query_lower = query.lower()
        answer_text = entry.get('answer', '').lower()
        question_text = entry.get('question', '').lower()
        
        # IISc M.Mgt specific terms
        domain_terms = {
            'iisc', 'm.mgt', 'management', 'bangalore', 'indian institute of science',
            'placement', 'ctc', 'salary', 'recruitment', 'interview', 'admission',
            'eligibility', 'curriculum', 'course', 'credit', 'semester', 'thesis',
            'project', 'internship', 'summer', 'faculty', 'research', 'campus',
            'hostel', 'accommodation', 'facility', 'library', 'gate', 'cat',
            'percentile', 'cutoff', 'marks', 'cgpa', 'engineering', 'technology'
        }
        
        # Check if query contains domain terms
        query_has_domain_terms = any(term in query_lower for term in domain_terms)
        
        # Check if content contains domain terms
        content_has_domain_terms = any(term in answer_text or term in question_text 
                                      for term in domain_terms)
        
        return query_has_domain_terms and content_has_domain_terms

    def _apply_diversity_filtering(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Apply diversity filtering to avoid similar results"""
        if not results:
            return results
        
        diverse_results = []
        used_content = []
        
        for result in results:
            # Check if this result is too similar to already selected ones
            is_similar = False
            current_content = result.content.get('answer', '').lower()
            current_words = set(current_content.split())
            
            for used_content_words in used_content:
                # Calculate similarity between current and used content
                if len(current_words) > 0 and len(used_content_words) > 0:
                    similarity = len(current_words.intersection(used_content_words)) / len(current_words.union(used_content_words))
                    if similarity > 0.6:  # If more than 60% similar, skip
                        is_similar = True
                        break
            
            if not is_similar:
                diverse_results.append(result)
                used_content.append(current_words)
                
                # Limit to reasonable number of diverse results
                if len(diverse_results) >= 5:
                    break
        
        return diverse_results

# ---------------------------
# Enhanced QA Agent with Qwen2.5 Model
# ---------------------------


class QAAgent:
    """Enhanced QA Agent with Qwen2.5 model and intelligent section extraction"""

    def __init__(self, device: int = 0, config: Optional[QAConfig] = None):
        self.config = config or QAConfig()
        self.device = device

        # Initialize AI generation flags
        self.use_ai_generation = False
        self.use_qa_pipeline = False  # Flag to track if using Q&A pipeline vs text generation
        self.generator = None
        self.tokenizer = None

        # Check if AI generation is disabled
        if not self.config.use_ai_generation:
            self.use_ai_generation = False
            logger.info(
                "AI generation disabled - using context extraction only")
            print("⚠️  AI generation disabled - using context extraction only")
            return

        # Initialize tokenizer for Q&A model
        try:
            # Use FLAN-T5 which is designed for instruction following and Q&A
            model_name = "google/flan-t5-small"  # Better for Q&A than DialoGPT
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Successfully loaded {model_name} tokenizer")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # Fallback to context extraction only
            self.use_ai_generation = False
            logger.info("Falling back to context extraction mode")
            return

        # Check device availability and set device properly
        if device >= 0 and torch.cuda.is_available():
            self.device = device
            device_name = torch.cuda.get_device_name(self.device)
            gpu_memory = torch.cuda.get_device_properties(
                self.device).total_memory / 1024**3
            logger.info(f"Using GPU: {device_name} ({gpu_memory:.1f}GB)")
            print(f"✓ Using GPU: {device_name} ({gpu_memory:.1f}GB)")
        else:
            self.device = -1
            logger.info("Using CPU for model inference")
            print("✓ Using CPU for model inference")

        # Set device map for pipeline initialization
        device_map = "auto" if self.device >= 0 else None
        torch_device = f"cuda:{self.device}" if self.device >= 0 else "cpu"
        logger.info(f"Device map: {device_map}, torch device: {torch_device}")

        # Initialize FLAN-T5 generation pipeline
        try:
            pipeline_kwargs = {
                "task": "text2text-generation",  # FLAN-T5 uses text2text-generation
                # FLAN-T5 model for better Q&A performance
                "model": "google/flan-t5-small",
                "tokenizer": self.tokenizer,
                # FLAN-T5 works well with shorter outputs
                "max_new_tokens": min(100, self.config.max_new_tokens),
                "temperature": 0.3,  # Lower temperature for more factual responses
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }

            # Add GPU-specific optimizations
            if self.device >= 0:
                pipeline_kwargs.update({
                    "device_map": "auto",  # Automatic GPU placement
                    "torch_dtype": torch.float16  # Use FP16 for faster inference
                })
                print("✓ Using GPU optimizations (FP16, auto device mapping)")
            else:
                pipeline_kwargs["device"] = "cpu"
                print("✓ Using CPU inference")

            self.generator = pipeline(**pipeline_kwargs)
            self.use_ai_generation = True
            logger.info(
                "Successfully initialized FLAN-T5-small model for text generation")
        except Exception as e:
            logger.error(f"Failed to initialize FLAN-T5-small model: {e}")
            # Try with an even smaller model as fallback
            try:
                logger.info("Trying with DistilBERT Q&A model...")
                pipeline_kwargs = {
                    "task": "question-answering",  # Use Q&A pipeline for DistilBERT
                    "model": "distilbert-base-cased-distilled-squad",  # Smaller Q&A model
                    "tokenizer": AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad"),
                    "max_answer_len": 100,
                    "return_full_text": False
                }

                if pipeline_kwargs["tokenizer"].pad_token is None:
                    pipeline_kwargs["tokenizer"].pad_token = pipeline_kwargs["tokenizer"].eos_token

                # Add GPU-specific optimizations
                if self.device >= 0:
                    pipeline_kwargs.update({
                        "device_map": "auto",
                        "torch_dtype": torch.float16
                    })
                else:
                    pipeline_kwargs["device"] = "cpu"

                self.generator = pipeline(**pipeline_kwargs)
                self.use_ai_generation = True
                self.use_qa_pipeline = True  # Flag to indicate we're using Q&A pipeline
                logger.info("Successfully initialized DistilBERT Q&A model")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback model: {e2}")
                # Final fallback: context extraction only
                self.use_ai_generation = False
                logger.info(
                    "Using context extraction mode only - no AI generation")
                print("⚠️  Using context extraction mode only (no AI generation)")
                print(
                    "   This will provide answers based on exact text matches from the knowledge base.")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=True))
        except:
            # Fallback: rough estimation
            return int(len(text.split()) * 1.3)

    def _truncate_context_intelligently(self, context: str, max_length: int = 800) -> str:
        """Truncate context while preserving complete sentences and important information"""
        if len(context) <= max_length:
            return context

        # Split into sentences
        sentences = []
        for sent in context.split('.'):
            sent = sent.strip()
            if sent:
                sentences.append(sent + '.')

        # Build truncated context by adding complete sentences
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence) <= max_length:
                truncated += sentence + " "
            else:
                break

        return truncated.strip()

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query to provide better instructions"""
        query_lower = query.lower()

        # Define query type patterns
        patterns = {
            'eligibility': ['eligibility', 'criteria', 'requirement', 'qualify', 'eligible'],
            'admission': ['admission', 'apply', 'application', 'deadline', 'cutoff', 'cat', 'gate'],
            'curriculum': ['course', 'curriculum', 'subject', 'credit', 'semester', 'elective'],
            'placement': ['placement', 'job', 'salary', 'ctc', 'company', 'recruit'],
            'fees': ['fee', 'cost', 'tuition', 'expense', 'scholarship', 'financial'],
            'campus': ['campus', 'hostel', 'accommodation', 'facility', 'life'],
            'interview': ['interview', 'selection', 'process', 'group discussion'],
            'duration': ['duration', 'year', 'time', 'long', 'period']
        }

        for query_type, keywords in patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type

        return 'general'

    def _get_domain_instructions(self, query_type: str) -> str:
        """Get domain-specific instructions based on query type"""
        instructions = {
            'eligibility': "Focus on specific degree requirements, marks/CGPA criteria, and any additional qualifications needed.",
            'admission': "Include specific percentiles, deadlines, application procedures, and required documents.",
            'curriculum': "Detail course names, credit structure, core vs elective courses, and any specialization options.",
            'placement': "Provide specific salary figures, company names, placement statistics, and role types.",
            'fees': "Include exact fee amounts, payment schedules, and any available financial aid information.",
            'campus': "Describe facilities, accommodation options, campus life, and student experiences.",
            'interview': "Explain the interview format, typical questions, preparation tips, and selection criteria.",
            'duration': "Specify exact time periods, semester structure, and any flexibility in program length.",
            'general': "Provide comprehensive information relevant to the IISc M.Mgt program."
        }

        return instructions.get(query_type, instructions['general'])

    def _analyze_query_intent(self, query: str) -> dict:
        """Analyze query to understand user intent and information needs"""
        intent = {
            'type': 'general',
            'specificity': 'general',  # specific, general, broad
            'information_type': 'factual',  # factual, procedural, comparative
            'urgency': 'normal'  # urgent, normal, exploratory
        }

        # Detect question type
        if any(word in query for word in ['what', 'which', 'who']):
            intent['information_type'] = 'factual'
        elif any(word in query for word in ['how', 'when', 'where']):
            intent['information_type'] = 'procedural'
        elif any(word in query for word in ['compare', 'difference', 'vs', 'versus']):
            intent['information_type'] = 'comparative'

        # Detect specificity
        if any(word in query for word in ['specific', 'exact', 'precise', 'detailed']):
            intent['specificity'] = 'specific'
        elif any(word in query for word in ['overview', 'general', 'about', 'tell me']):
            intent['specificity'] = 'broad'

        # Detect urgency/importance
        if any(word in query for word in ['urgent', 'important', 'deadline', 'last date']):
            intent['urgency'] = 'urgent'
        elif any(word in query for word in ['explore', 'learn', 'understand', 'know more']):
            intent['urgency'] = 'exploratory'

        # Detect primary topic
        intent['type'] = self._detect_query_type(query)

        return intent

    def _create_qwen_prompt(self, query: str, context: str) -> str:
        """Create an instruction prompt for FLAN-T5"""
        # Truncate context to a reasonable length for FLAN-T5
        truncated_context = self._truncate_context_intelligently(
            context, max_length=500)

        # Create an instruction-style prompt that FLAN-T5 can handle well
        return f"""Answer the following question based on the provided context. Be specific and factual.

Context: {truncated_context}

Question: {query}

Answer:"""

    def _clean_qwen_output(self, text: str) -> str:
        """Clean FLAN-T5 model output with minimal processing to preserve content"""
        if not text:
            return ""

        # FLAN-T5 typically produces cleaner output, minimal cleaning needed
        text = text.replace("<pad>", "").replace(
            "</s>", "").replace("<unk>", "")

        # Remove only leading/trailing whitespace, preserve internal formatting
        text = text.strip()

        # FLAN-T5 usually produces complete sentences, but add period if needed
        if text and len(text) > 3 and not text.endswith(('.', '!', '?', ':', ';')):
            # Check if it looks like a complete thought before adding period
            if not text.endswith(('...', '--', '-')):
                text += '.'

        return text

    def _validate_answer_grounding(self, answer: str, context: str) -> bool:
        """Enhanced validation to check if the answer is reasonable and well-grounded"""
        if not answer or not context:
            return False

        answer_lower = answer.lower()
        context_lower = context.lower()

        # Reject answers that are clearly problematic
        problematic_phrases = [
            "i am an ai", "i cannot", "as an ai", "i don't know",
            "sorry, i cannot", "i'm sorry, but i don't have access to",
            "i don't have enough information", "i cannot provide specific information",
            "i'm not sure", "i'm unable to", "i can't help", "i don't have access",
            "based on the information provided", "according to the context", "the context shows"
        ]

        # If answer contains problematic phrases, reject
        if any(phrase in answer_lower for phrase in problematic_phrases):
            return False

        # Check basic quality metrics
        if len(answer.split()) < 5 or len(answer) < 20:
            return False

        # Check for basic sentence structure
        if not any(char in answer for char in ['.', '!', '?', ':', ';']):
            return False

        # Enhanced grounding check - verify answer relates to context
        grounding_score = self._calculate_grounding_score(
            answer_lower, context_lower)

        # Check for domain relevance (IISc M.Mgt specific)
        domain_relevance = self._check_domain_relevance(answer_lower)

        # Check for numerical information (often indicates specific, useful answers)
        has_numerical_info = self._has_numerical_information(answer_lower)

        # Check for specific, actionable information
        has_specific_info = self._has_specific_information(answer_lower)

        # More stringent validation for better quality answers:
        # 1. Good grounding score
        # 2. Domain relevant content
        # 3. Contains specific, actionable information
        # 4. Contains numerical information with decent grounding

        if grounding_score >= 0.25:  # Higher threshold for better quality
            return True
        elif domain_relevance and has_specific_info:
            return True
        elif has_numerical_info and grounding_score >= 0.15:  # Higher threshold for numerical info
            return True
        elif len(answer.split()) >= 12 and grounding_score >= 0.2:  # Longer answers need better grounding
            return True
        else:
            return False

    def _validate_answer_grounding_with_details(self, answer: str, context: str) -> tuple[bool, str]:
        """Enhanced validation with detailed feedback for debugging"""
        if not answer or not context:
            return False, "Empty answer or context"

        answer_lower = answer.lower()
        context_lower = context.lower()

        # Check for problematic phrases
        problematic_phrases = [
            "i am an ai", "i cannot", "as an ai", "i don't know",
            "sorry, i cannot", "i'm sorry, but i don't have access to",
            "i don't have enough information", "i cannot provide specific information",
            "i'm not sure", "i'm unable to", "i can't help", "i don't have access"
        ]

        for phrase in problematic_phrases:
            if phrase in answer_lower:
                return False, f"Contains problematic phrase: '{phrase}'"

        # Check basic quality metrics
        word_count = len(answer.split())
        if word_count < 3:
            return False, f"Too short: {word_count} words"

        if len(answer) < 15:
            return False, f"Too short: {len(answer)} characters"

        # Check for basic sentence structure
        if not any(char in answer for char in ['.', '!', '?', ':', ';']):
            return False, "No proper sentence ending punctuation"

        # Calculate scores
        grounding_score = self._calculate_grounding_score(
            answer_lower, context_lower)
        domain_relevance = self._check_domain_relevance(answer_lower)
        has_numerical_info = self._has_numerical_information(answer_lower)

        # Apply lenient validation logic
        if grounding_score >= 0.15:
            return True, f"Good grounding score: {grounding_score:.3f}"
        elif domain_relevance:
            return True, f"Domain relevant (grounding: {grounding_score:.3f})"
        elif has_numerical_info and grounding_score >= 0.05:
            return True, f"Has numerical info (grounding: {grounding_score:.3f})"
        elif word_count >= 8 and grounding_score >= 0.08:
            return True, f"Long answer with minimal grounding: {word_count} words (grounding: {grounding_score:.3f})"
        else:
            return False, f"Low grounding score: {grounding_score:.3f}, domain_relevant: {domain_relevance}, has_numbers: {has_numerical_info}, words: {word_count}"

    def _calculate_grounding_score(self, answer: str, context: str) -> float:
        """Calculate how well the answer is grounded in the provided context with improved scoring"""
        answer_words = set(answer.split())
        context_words = set(context.split())

        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

        answer_content_words = answer_words - stop_words
        context_content_words = context_words - stop_words

        if not answer_content_words:
            return 0.0

        # Calculate multiple types of overlap for better scoring
        exact_overlap = len(
            answer_content_words.intersection(context_content_words))

        # Also check for partial word matches (for variations like "eligibility" vs "eligible")
        partial_matches = 0
        for answer_word in answer_content_words:
            for context_word in context_content_words:
                # Check if words share a common root (at least 4 characters)
                if len(answer_word) >= 4 and len(context_word) >= 4:
                    if answer_word[:4] == context_word[:4] or answer_word[-4:] == context_word[-4:]:
                        partial_matches += 0.5
                        break

        # Calculate weighted score
        total_matches = exact_overlap + partial_matches
        base_score = total_matches / len(answer_content_words)

        # Boost score for answers with important domain terms
        domain_boost = 0.0
        important_terms = {'iisc', 'management', 'eligibility',
                           'admission', 'placement', 'ctc', 'percentile', 'cgpa', 'credits'}
        for term in important_terms:
            if term in answer.lower() and term in context.lower():
                domain_boost += 0.05

        return min(1.0, base_score + domain_boost)

    def _calculate_content_relevance(self, content: dict, query: str) -> float:
        """Calculate how relevant a content piece is to the query"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Get content text
        answer_text = content.get('answer', '').lower()
        question_text = content.get('question', '').lower()
        
        # Calculate word overlap
        answer_words = set(answer_text.split())
        question_words = set(question_text.split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = query_words - stop_words
        answer_words = answer_words - stop_words
        question_words = question_words - stop_words
        
        if not query_words:
            return 0.0
        
        # Calculate overlap scores
        answer_overlap = len(query_words.intersection(answer_words)) / len(query_words)
        question_overlap = len(query_words.intersection(question_words)) / len(query_words)
        
        # Weight question overlap more heavily as it's more indicative of relevance
        relevance_score = (question_overlap * 0.7) + (answer_overlap * 0.3)
        
        return min(1.0, relevance_score)

    def _assess_answer_quality(self, answer: str, query: str) -> float:
        """Assess the overall quality of an answer"""
        if not answer or not query:
            return 0.0
        
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # Length quality (not too short, not too long)
        word_count = len(answer.split())
        if 10 <= word_count <= 100:
            length_score = 1.0
        elif 5 <= word_count <= 150:
            length_score = 0.8
        else:
            length_score = 0.5
        
        # Specificity score
        specificity_score = 0.0
        if self._has_specific_information(answer):
            specificity_score += 0.3
        if self._has_numerical_information(answer):
            specificity_score += 0.2
        if self._check_domain_relevance(answer):
            specificity_score += 0.3
        
        # Coherence score (basic sentence structure)
        coherence_score = 0.0
        if any(char in answer for char in ['.', '!', '?']):
            coherence_score += 0.2
        if len(answer.split()) >= 5:
            coherence_score += 0.2
        if not any(phrase in answer_lower for phrase in ['i cannot', 'i don\'t know', 'sorry']):
            coherence_score += 0.3
        
        # Query relevance score
        query_relevance = self._calculate_grounding_score(answer_lower, query_lower)
        
        # Combine scores
        total_score = (
            length_score * 0.2 +
            specificity_score * 0.4 +
            coherence_score * 0.2 +
            query_relevance * 0.2
        )
        
        return min(1.0, total_score)

    def _check_domain_relevance(self, answer: str) -> bool:
        """Check if answer contains domain-relevant terms for IISc M.Mgt"""
        domain_terms = {
            'iisc', 'management', 'mgt', 'm.mgt', 'bangalore', 'indian institute of science',
            'engineering', 'technology', 'analytics', 'data science', 'placement', 'ctc',
            'gate', 'cat', 'percentile', 'eligibility', 'admission', 'curriculum', 'credits',
            'semester', 'internship', 'thesis', 'project', 'campus', 'hostel', 'fee'
        }

        answer_words = set(answer.split())
        return len(answer_words.intersection(domain_terms)) >= 1

    def _has_numerical_information(self, answer: str) -> bool:
        """Check if answer contains numerical information that indicates specificity"""
        import re

        # Look for various types of numerical information
        numerical_patterns = [
            r'\d+%',  # Percentages
            r'\d+\.\d+',  # Decimal numbers
            r'₹[\d,]+',  # Currency amounts
            r'\d+\s*lpa',  # Salary in LPA
            r'\d+\s*lakhs?',  # Lakhs
            r'\d+\s*crores?',  # Crores
            r'\d+\s*years?',  # Years
            r'\d+\s*months?',  # Months
            r'\d+\s*credits?',  # Credits
            r'\d+\s*semesters?',  # Semesters
            r'\d+\s*percentile',  # Percentiles
            r'\d+\s*marks?',  # Marks
            r'\d+\s*cgpa',  # CGPA
            r'\d{4}',  # Years (4 digits)
        ]

        return any(re.search(pattern, answer, re.IGNORECASE) for pattern in numerical_patterns)

    def _has_specific_information(self, answer: str) -> bool:
        """Check if answer contains specific, actionable information"""
        answer_lower = answer.lower()
        
        # Look for specific terms that indicate actionable information
        specific_indicators = [
            'specific', 'exact', 'precise', 'detailed', 'particular',
            'include', 'consist', 'comprise', 'contain', 'involve',
            'require', 'need', 'must', 'should', 'typically',
            'usually', 'generally', 'commonly', 'often', 'frequently',
            'available', 'offered', 'provided', 'facilitated', 'supported',
            'through', 'via', 'using', 'with', 'by', 'during', 'while',
            'access', 'participate', 'enroll', 'apply', 'submit',
            'deadline', 'cutoff', 'percentile', 'score', 'marks',
            'company', 'firm', 'organization', 'institution', 'university',
            'faculty', 'professor', 'researcher', 'student', 'graduate'
        ]
        
        # Check for specific IISc M.Mgt terms
        domain_specific_terms = [
            'iisc', 'm.mgt', 'management', 'bangalore', 'indian institute',
            'placement', 'ctc', 'salary', 'recruitment', 'interview',
            'curriculum', 'course', 'credit', 'semester', 'thesis',
            'project', 'internship', 'summer', 'faculty', 'research',
            'campus', 'hostel', 'accommodation', 'facility', 'library'
        ]
        
        # Count specific indicators
        specific_count = sum(1 for term in specific_indicators if term in answer_lower)
        domain_count = sum(1 for term in domain_specific_terms if term in answer_lower)
        
        # Answer is specific if it has multiple specific indicators or domain-specific terms
        return specific_count >= 3 or domain_count >= 2

    def _extract_relevant_sentences(self, query: str, text: str, max_sentences: int = 4) -> str:
        """Enhanced sentence extraction with better relevance scoring and context awareness"""
        if not text:
            return ""

        query_lower = query.lower()
        query_words = set(word.lower()
                          for word in query.split() if len(word) > 2)
        query_type = self._detect_query_type(query)

        # Split text into sentences more intelligently
        sentences = self._split_into_sentences(text)

        if not sentences:
            return ""

        # Score each sentence with enhanced algorithm
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self._calculate_sentence_relevance_score(
                sentence, query_words, query_type, query_lower, i, len(
                    sentences)
            )
            sentence_scores.append((sentence, score, i))

        # Sort by score and select best sentences
        sentence_scores.sort(key=lambda x: (-x[1], x[2]))

        # Select and synthesize top sentences with better coherence
        selected_sentences = self._select_and_synthesize_sentences_improved(
            sentence_scores, max_sentences, query_type, query_lower
        )

        if selected_sentences:
            result = self._format_final_answer_improved(selected_sentences, query_type)
            return result

        return ""

    def _split_into_sentences(self, text: str) -> list:
        """Split text into meaningful sentences"""
        sentences = []

        # Split on multiple sentence endings
        import re
        sentence_endings = re.split(r'[.!?]+', text)

        for sent in sentence_endings:
            sent = sent.strip()
            # Filter out very short or meaningless fragments
            if len(sent) > 15 and len(sent.split()) > 3:
                sentences.append(sent)

        return sentences

    def _calculate_sentence_relevance_score(self, sentence: str, query_words: set,
                                            query_type: str, query_lower: str,
                                            position: int, total_sentences: int) -> float:
        """Calculate comprehensive relevance score for a sentence"""
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())

        # Base scoring
        exact_matches = sum(
            1 for word in query_words if word in sentence_lower)
        word_overlap = len(query_words.intersection(sentence_words))

        # Position bonus (earlier sentences often more important)
        position_bonus = (total_sentences - position) / total_sentences * 0.3

        # Query type specific scoring
        type_bonus = self._get_query_type_bonus(sentence_lower, query_type)

        # Numerical information bonus (important for specific queries)
        numerical_bonus = self._get_numerical_bonus(sentence_lower, query_type)

        # Sentence quality score
        quality_score = self._assess_sentence_quality(sentence)

        # Combine all scores
        total_score = (
            exact_matches * 3.0 +           # Exact word matches are very important
            word_overlap * 2.0 +            # Word overlap is important
            type_bonus * 2.5 +              # Query type relevance
            numerical_bonus * 1.5 +         # Numerical information
            position_bonus +                # Position in text
            quality_score                   # Overall sentence quality
        )

        return total_score

    def _get_query_type_bonus(self, sentence: str, query_type: str) -> float:
        """Get bonus score based on query type specific keywords"""
        type_keywords = {
            'eligibility': ['degree', 'bachelor', 'engineering', 'marks', 'cgpa', 'minimum', 'required', 'qualification'],
            'placement': ['ctc', 'salary', 'companies', 'placed', 'offers', 'average', 'highest', 'package', 'lpa'],
            'interview': ['interview', 'questions', 'math', 'probability', 'discussion', 'selection', 'process'],
            'campus': ['campus', 'classes', 'facilities', 'accommodation', 'hostel', 'life', 'sports'],
            'curriculum': ['courses', 'credits', 'subjects', 'electives', 'semester', 'core', 'stream'],
            'admission': ['percentile', 'cutoff', 'cat', 'gate', 'application', 'deadline', 'admission'],
            'fees': ['fee', 'cost', 'tuition', 'expense', 'scholarship', 'financial', 'payment'],
            'duration': ['year', 'duration', 'time', 'semester', 'period', 'full-time']
        }

        keywords = type_keywords.get(query_type, [])
        matches = sum(1 for keyword in keywords if keyword in sentence)
        return matches * 0.5

    def _get_numerical_bonus(self, sentence: str, query_type: str) -> float:
        """Bonus for sentences containing numerical information relevant to query type"""
        import re

        # Look for different types of numbers based on query type
        if query_type in ['placement', 'fees']:
            # Look for salary/fee amounts
            if re.search(r'₹[\d,]+|lpa|\d+\s*lakhs?', sentence):
                return 1.0
        elif query_type == 'admission':
            # Look for percentiles, cutoffs
            if re.search(r'\d+\.\d+|\d+%|\d+\s*percentile', sentence):
                return 1.0
        elif query_type == 'curriculum':
            # Look for credit numbers
            if re.search(r'\d+\s*credits?|\d+\s*courses?', sentence):
                return 1.0
        elif query_type == 'duration':
            # Look for time periods
            if re.search(r'\d+\s*years?|\d+\s*semesters?', sentence):
                return 1.0

        # General numerical information
        if re.search(r'\d+', sentence):
            return 0.3

        return 0.0

    def _assess_sentence_quality(self, sentence: str) -> float:
        """Assess the overall quality and informativeness of a sentence"""
        # Length bonus (not too short, not too long)
        length = len(sentence.split())
        if 8 <= length <= 25:
            length_bonus = 0.3
        elif 5 <= length <= 35:
            length_bonus = 0.1
        else:
            length_bonus = 0.0

        # Completeness bonus (has proper sentence structure)
        completeness_bonus = 0.2 if sentence.strip().endswith(('.', '!', '?')) else 0.0

        # Information density (avoid very generic sentences)
        info_words = ['specific', 'include', 'such as',
                      'for example', 'approximately', 'exactly']
        info_bonus = 0.1 if any(word in sentence.lower()
                                for word in info_words) else 0.0

        return length_bonus + completeness_bonus + info_bonus

    def _select_and_synthesize_sentences(self, sentence_scores: list, max_sentences: int, query_type: str) -> list:
        """Select and potentially synthesize the best sentences"""
        selected = []
        used_content = []  # Use list instead of set to store word sets

        for sentence, score, _ in sentence_scores:
            if len(selected) >= max_sentences:
                break

            if score <= 0:
                continue

            # Avoid very similar sentences
            sentence_words = set(sentence.lower().split())
            if not any(len(sentence_words.intersection(used)) > len(sentence_words) * 0.7
                       for used in used_content):
                selected.append(sentence)
                # Append to list instead of add to set
                used_content.append(sentence_words)

        return selected

    def _select_and_synthesize_sentences_improved(self, sentence_scores: list, max_sentences: int, query_type: str, query_lower: str) -> list:
        """Improved sentence selection with better coherence and relevance"""
        selected = []
        used_content = []
        
        # First pass: select high-scoring sentences
        for sentence, score, _ in sentence_scores:
            if len(selected) >= max_sentences:
                break
                
            if score <= 0:
                continue
                
            # Avoid very similar sentences
            sentence_words = set(sentence.lower().split())
            if not any(len(sentence_words.intersection(used)) > len(sentence_words) * 0.6
                       for used in used_content):
                selected.append(sentence)
                used_content.append(sentence_words)
        
        # If we have enough sentences, return them
        if len(selected) >= 2:
            return selected
            
        # Second pass: if we need more sentences, be more lenient
        for sentence, score, _ in sentence_scores:
            if len(selected) >= max_sentences:
                break
                
            if score <= 0:
                continue
                
            # Check if this sentence is already selected
            if sentence in selected:
                continue
                
            # More lenient similarity check
            sentence_words = set(sentence.lower().split())
            if not any(len(sentence_words.intersection(used)) > len(sentence_words) * 0.8
                       for used in used_content):
                selected.append(sentence)
                used_content.append(sentence_words)
        
        return selected

    def _format_final_answer_improved(self, sentences: list, query_type: str) -> str:
        """Format the final answer with better coherence and structure"""
        if not sentences:
            return ""
        
        # Clean and prepare sentences
        cleaned_sentences = []
        for sentence in sentences:
            sent = sentence.strip()
            if sent:
                # Ensure proper sentence ending
                if not sent.endswith(('.', '!', '?')):
                    sent += '.'
                cleaned_sentences.append(sent)
        
        if not cleaned_sentences:
            return ""
        
        # Join sentences with proper spacing
        result = ' '.join(cleaned_sentences)
        
        # Clean up any double periods or spaces
        result = result.replace('..', '.')
        result = result.replace('  ', ' ')
        
        # Add query-type specific formatting
        if query_type in ['placement', 'salary', 'ctc']:
            # For placement questions, ensure numerical information is clear
            if any(char.isdigit() for char in result):
                result = result.replace('₹', '₹').replace('LPA', ' LPA').replace('lpa', ' LPA')
        
        elif query_type in ['eligibility', 'admission']:
            # For admission questions, ensure requirements are clear
            if 'required' in result.lower() or 'need' in result.lower():
                result = result.replace(' - ', '. ').replace('- ', '. ')
        
        elif query_type in ['curriculum', 'course']:
            # For curriculum questions, ensure course structure is clear
            if 'include' in result.lower() or 'consist' in result.lower():
                result = result.replace(' - ', '. ').replace('- ', '. ')
        
        return result.strip()

    def _format_final_answer(self, sentences: list) -> str:
        """Format the final answer from selected sentences"""
        if not sentences:
            return ""

        # Join sentences properly
        result = '. '.join(sent.strip() for sent in sentences)

        # Ensure proper ending
        if not result.endswith(('.', '!', '?')):
            result += '.'

        # Clean up any double periods
        result = result.replace('..', '.')

        return result

    def _generate_answer_with_ai(self, query: str, context: str) -> str:
        """Generate answer using AI model"""
        try:
            if not self.use_ai_generation:
                return ""

            # Create prompt
            prompt = self._create_qwen_prompt(query, context)

            # Generate response with better parameters for FLAN-T5
            logger.debug(f"Generating with prompt length: {len(prompt)}")
            logger.debug(f"Prompt preview: '{prompt[:100]}...'")

            response = self.generator(
                prompt,
                max_new_tokens=80,  # Good length for FLAN-T5 factual responses
                temperature=0.3,  # Lower temperature for more factual responses
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Avoid repetition
                # Note: return_full_text not supported by text2text-generation pipeline
            )

            logger.debug(f"Generator returned {len(response)} responses")

            # Extract and clean the generated text
            if response and len(response) > 0:
                generated_text = response[0]['generated_text']
                logger.debug(f"Raw AI response length: {len(generated_text)}")
                logger.debug(
                    f"Raw AI response content: '{generated_text[:200]}...'")

                # More intelligent prompt removal - be conservative
                original_length = len(generated_text)

                # Try different approaches to remove prompt without losing content
                if prompt in generated_text:
                    # Method 1: Direct prompt removal
                    generated_text = generated_text[len(prompt):]
                    logger.debug(
                        f"Removed exact prompt match, length: {len(generated_text)}")
                elif generated_text.startswith(prompt[:50]):  # Partial match
                    # Method 2: Find where the actual response starts
                    # Look for common response indicators
                    response_indicators = [
                        "ANSWER:", "Answer:", "answer:", "\n", ":", "A:"]
                    for indicator in response_indicators:
                        if indicator in generated_text:
                            idx = generated_text.find(indicator)
                            if idx > 0:
                                generated_text = generated_text[idx +
                                                                len(indicator):].strip()
                                logger.debug(
                                    f"Found response after '{indicator}', length: {len(generated_text)}")
                                break
                else:
                    logger.debug(
                        "No prompt removal needed - keeping full response")

                logger.debug(
                    f"After prompt processing: {len(generated_text)} chars (was {original_length})")

                logger.debug(
                    f"Text after prompt removal: '{generated_text[:200]}...'")

                cleaned_text = self._clean_qwen_output(generated_text)
                logger.debug(f"Cleaned text length: {len(cleaned_text)}")
                logger.debug(f"Cleaned text content: '{cleaned_text}'")

                # Safety check: if cleaning removed everything, use the raw generated text
                if not cleaned_text or len(cleaned_text.strip()) == 0:
                    if generated_text and len(generated_text.strip()) > 0:
                        logger.warning(
                            "Cleaning removed all content, using raw generated text")
                        cleaned_text = generated_text.strip()
                        # Add basic punctuation if needed
                        if cleaned_text and not cleaned_text.endswith(('.', '!', '?')):
                            cleaned_text += '.'
                    else:
                        logger.warning(
                            "AI generated empty response after cleaning")
                        logger.debug(
                            f"Original response was: '{response[0]['generated_text']}'")
                        logger.debug(f"Prompt was: '{prompt}'")
                        return ""

                # Validate the answer with detailed logging
                is_valid, validation_details = self._validate_answer_grounding_with_details(
                    cleaned_text, context)
                if is_valid:
                    logger.info(
                        f"AI answer passed validation: {validation_details}")
                    return cleaned_text
                else:
                    logger.warning(
                        f"Generated answer failed validation: {validation_details}. Falling back to context extraction")
                    return ""
            else:
                logger.warning("No response generated from AI model")
                return ""

        except Exception as e:
            logger.error(f"Error generating AI answer: {e}")
            return ""

    def _extract_direct_answer(self, query: str, search_results: List) -> str:
        """Enhanced answer extraction with intelligent context fusion and ranking"""
        if not search_results:
            return "I couldn't find relevant information to answer your question."

        # Analyze query to determine best extraction strategy
        query_intent = self._analyze_query_intent(query.lower())

        # Rank and filter search results for better context selection
        ranked_results = self._rank_search_results_for_query(
            search_results, query, query_intent)

        if not ranked_results:
            return "I found some information but it doesn't seem relevant to your specific question."

        # Get intelligently selected contexts
        selected_contexts = self._select_best_contexts(
            ranked_results, query_intent)

        # Try AI generation first if available and appropriate
        if self.use_ai_generation and selected_contexts:
            combined_context = self._fuse_contexts_intelligently(
                selected_contexts, query_intent)
            ai_answer = self._generate_answer_with_ai(query, combined_context)
            if ai_answer:
                return ai_answer

        # Enhanced fallback to sentence extraction with context fusion
        return self._extract_answer_from_contexts(query, selected_contexts, query_intent)

    def _rank_search_results_for_query(self, search_results: List, query: str, query_intent: dict) -> List:
        """Rank search results based on query-specific relevance"""
        ranked_results = []

        for result in search_results:
            # Calculate query-specific relevance score
            relevance_score = self._calculate_query_specific_relevance(
                result, query, query_intent)

            # Only include results above minimum threshold
            if relevance_score > 0.1:
                ranked_results.append((result, relevance_score))

        # Sort by relevance score
        ranked_results.sort(key=lambda x: x[1], reverse=True)

        return [result for result, score in ranked_results]

    def _calculate_query_specific_relevance(self, result, query: str, query_intent: dict) -> float:
        """Calculate how relevant a search result is to the specific query"""
        content = result.content
        answer_text = content.get('answer', '').lower()
        question_text = content.get('question', '').lower()

        # Base confidence from search
        base_score = result.confidence

        # Query word overlap bonus
        query_words = set(query.lower().split())
        answer_words = set(answer_text.split())
        question_words = set(question_text.split())

        answer_overlap = len(query_words.intersection(
            answer_words)) / len(query_words) if query_words else 0
        question_overlap = len(query_words.intersection(
            question_words)) / len(query_words) if query_words else 0

        # Query type specific bonus
        type_bonus = self._get_content_type_relevance(
            answer_text, query_intent['type'])

        # Information type bonus (factual vs procedural)
        info_type_bonus = self._get_information_type_relevance(
            answer_text, query_intent['information_type'])

        # Combine scores
        total_score = (
            base_score * 0.4 +
            answer_overlap * 0.3 +
            question_overlap * 0.1 +
            type_bonus * 0.15 +
            info_type_bonus * 0.05
        )

        return total_score

    def _get_content_type_relevance(self, content: str, query_type: str) -> float:
        """Get relevance bonus based on content matching query type"""
        type_indicators = {
            'eligibility': ['degree', 'bachelor', 'marks', 'cgpa', 'minimum', 'required'],
            'admission': ['application', 'deadline', 'cutoff', 'percentile', 'cat', 'gate'],
            'placement': ['salary', 'ctc', 'companies', 'average', 'highest', 'placed'],
            'curriculum': ['courses', 'credits', 'semester', 'subjects', 'core', 'electives'],
            'fees': ['fee', 'cost', 'tuition', 'payment', 'scholarship'],
            'campus': ['campus', 'hostel', 'facilities', 'accommodation', 'life'],
            'interview': ['interview', 'selection', 'process', 'questions', 'discussion']
        }

        indicators = type_indicators.get(query_type, [])
        matches = sum(1 for indicator in indicators if indicator in content)

        return min(1.0, matches * 0.2)

    def _get_information_type_relevance(self, content: str, info_type: str) -> float:
        """Get relevance bonus based on information type match"""
        if info_type == 'factual':
            # Look for specific facts, numbers, names
            import re
            if re.search(r'\d+|₹|%|specific|exactly|precisely', content):
                return 0.3
        elif info_type == 'procedural':
            # Look for process words
            if any(word in content for word in ['how', 'process', 'steps', 'procedure', 'apply', 'submit']):
                return 0.3
        elif info_type == 'comparative':
            # Look for comparison words
            if any(word in content for word in ['compare', 'versus', 'difference', 'better', 'higher', 'lower']):
                return 0.3

        return 0.0

    def _select_best_contexts(self, ranked_results: List, query_intent: dict) -> List[str]:
        """Select the best contexts based on query intent and diversity"""
        contexts = []
        max_contexts = 3 if query_intent['specificity'] == 'specific' else 4

        for i, result in enumerate(ranked_results[:max_contexts]):
            answer_text = result.content.get('answer', '')
            if answer_text and len(answer_text.strip()) > 20:
                contexts.append(answer_text)

        return contexts

    def _fuse_contexts_intelligently(self, contexts: List[str], query_intent: dict) -> str:
        """Intelligently fuse multiple contexts for better AI generation"""
        if not contexts:
            return ""

        if len(contexts) == 1:
            return contexts[0]

        # For specific queries, prioritize the most relevant context
        if query_intent['specificity'] == 'specific':
            # Take the best context and supplement with key info from others
            primary_context = contexts[0]
            supplementary_info = []

            for context in contexts[1:]:
                # Extract key numerical or specific information
                import re
                numbers = re.findall(
                    r'\d+(?:\.\d+)?(?:%|₹|lpa|lakhs?|years?|credits?)', context)
                if numbers:
                    supplementary_info.extend(numbers)

            if supplementary_info:
                return f"{primary_context} Additional details: {', '.join(set(supplementary_info))}"
            else:
                return primary_context
        else:
            # For general queries, combine contexts more comprehensively
            return " ".join(contexts)

    def _extract_answer_from_contexts(self, query: str, contexts: List[str], query_intent: dict) -> str:
        """Extract answer from contexts when AI generation is not available"""
        if not contexts:
            return "I found some information but couldn't extract a specific answer."

        # Use the enhanced sentence extraction on the best context
        primary_context = contexts[0]

        # Adjust extraction parameters based on query intent
        if query_intent['specificity'] == 'specific':
            max_sentences = 2
        elif query_intent['specificity'] == 'broad':
            max_sentences = 4
        else:
            max_sentences = 3

        relevant_section = self._extract_relevant_sentences(
            query, primary_context, max_sentences=max_sentences)

        if relevant_section:
            return relevant_section
        else:
            # Final fallback with better sentence selection
            sentences = primary_context.split('.')
            # Select sentences that contain query-relevant terms
            query_words = set(query.lower().split())
            relevant_sentences = []

            for sentence in sentences[:5]:  # Check first 5 sentences
                sentence = sentence.strip()
                if len(sentence) > 15:
                    sentence_words = set(sentence.lower().split())
                    if query_words.intersection(sentence_words):
                        relevant_sentences.append(sentence)
                        if len(relevant_sentences) >= 2:
                            break

            if relevant_sentences:
                return '. '.join(relevant_sentences) + '.'
            else:
                # Last resort: return first meaningful sentence
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:
                        return sentence + '.'

                return "I found some information but couldn't extract a specific answer."

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

            # Improved confidence calculation with better accuracy
            # Weight by position and source quality
            weighted_confidence = 0
            total_weight = 0
            
            for i, result in enumerate(search_results[:3]):
                # Position weight (first results are more important)
                position_weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33...
                
                # Source quality weight
                source_type = result.content.get("metadata", {}).get("source", "unknown")
                source_weight = 1.0
                if source_type == "json":
                    source_weight = 1.3  # Higher weight for curated FAQ data
                elif source_type == "web":
                    source_weight = 0.9  # Lower weight for web content
                
                # Content relevance weight
                content_relevance = self._calculate_content_relevance(result.content, query)
                relevance_weight = 1.0 + (content_relevance * 0.5)  # Boost up to 50%
                
                # Combined weight
                combined_weight = position_weight * source_weight * relevance_weight
                weighted_confidence += result.confidence * combined_weight
                total_weight += combined_weight

            avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0

            # Quality-based confidence adjustments
            good_matches = [r for r in search_results if r.confidence > 0.7]
            json_sources = [r for r in search_results if r.content.get(
                "metadata", {}).get("source") == "json"]
            
            # Check answer quality
            answer_quality = self._assess_answer_quality(answer, query)
            
            # Apply quality-based adjustments
            if answer_quality > 0.8:
                avg_confidence = min(1.0, avg_confidence * 1.1)  # Boost for high-quality answers
            elif answer_quality < 0.4:
                avg_confidence = max(0.0, avg_confidence * 0.8)  # Reduce for low-quality answers
            
            if len(good_matches) >= 2:
                avg_confidence = min(1.0, avg_confidence * 1.15)
            elif len(json_sources) >= 1:
                avg_confidence = min(1.0, avg_confidence * 1.1)

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
        min_confidence=getattr(args, 'min_confidence', 0.1),
        use_ai_generation=not getattr(args, 'no_ai_generation', False)
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
    parser.add_argument('--no_ai_generation', action='store_true',
                        help='Disable AI model loading and use only context extraction')

    # Output options
    parser.add_argument('--max_print', type=int, default=1,
                        help='Number of answers to print')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')

    args = parser.parse_args()

    # If neither --faq nor --url is provided, load all sources from sources.json
    if not args.faq and not args.url:
        logger.info(
            "No --faq or --url provided. Loading all sources from sources.json for agentic mode.")
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
                            logger.warning(
                                f"Failed to add source {src.get('name', 'Unknown')}: {e}")
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
