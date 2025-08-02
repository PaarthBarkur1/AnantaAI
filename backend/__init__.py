"""
AnantaAI Backend Package

This package contains the core components for the AnantaAI question-answering system.
"""

# Version information
__version__ = "2.0.0"
__author__ = "AnantaAI Team"

# Import main components for easy access
try:
    from .qna import (
        ContextAgent,
        QAAgent,
        QAConfig,
        JSONContentSource,
        WebContentSource,
        SearchResult
    )
    from .embedding_config import EmbeddingModelManager
    from .webscrapper import WebScraper
    from .faq_data import FAQ_DATA
except ImportError:
    # Fallback imports for direct execution
    pass

__all__ = [
    'ContextAgent',
    'QAAgent', 
    'QAConfig',
    'JSONContentSource',
    'WebContentSource',
    'SearchResult',
    'EmbeddingModelManager',
    'WebScraper',
    'FAQ_DATA'
]
