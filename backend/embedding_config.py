"""
Embedding Model Configuration for AnantaAI
This module provides configuration and utilities for managing different embedding models.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model"""
    name: str
    dimensions: int
    description: str
    performance: str  # excellent, very good, good, fair
    size: str  # large, medium, small
    recommended_use: str
    memory_usage: str  # high, medium, low
    speed: str  # fast, medium, slow


class EmbeddingModelManager:
    """Manager for embedding model configurations"""
    
    AVAILABLE_MODELS = {
        "BAAI/bge-large-en-v1.5": EmbeddingModelConfig(
            name="BAAI/bge-large-en-v1.5",
            dimensions=1024,
            description="State-of-the-art for retrieval tasks",
            performance="excellent",
            size="large",
            recommended_use="Production systems requiring highest accuracy",
            memory_usage="high",
            speed="medium"
        ),
        "intfloat/e5-large-v2": EmbeddingModelConfig(
            name="intfloat/e5-large-v2",
            dimensions=1024,
            description="Excellent for diverse tasks and multilingual support",
            performance="excellent",
            size="large",
            recommended_use="Multilingual or diverse domain applications",
            memory_usage="high",
            speed="medium"
        ),
        "sentence-transformers/all-mpnet-base-v2": EmbeddingModelConfig(
            name="sentence-transformers/all-mpnet-base-v2",
            dimensions=768,
            description="Best overall performance for general tasks",
            performance="very good",
            size="medium",
            recommended_use="General purpose applications",
            memory_usage="medium",
            speed="medium"
        ),
        "sentence-transformers/all-MiniLM-L12-v2": EmbeddingModelConfig(
            name="sentence-transformers/all-MiniLM-L12-v2",
            dimensions=384,
            description="Good balance of speed and performance",
            performance="good",
            size="small",
            recommended_use="Resource-constrained environments",
            memory_usage="low",
            speed="fast"
        ),
        "all-MiniLM-L6-v2": EmbeddingModelConfig(
            name="all-MiniLM-L6-v2",
            dimensions=384,
            description="Fast and lightweight (legacy)",
            performance="fair",
            size="small",
            recommended_use="Development and testing",
            memory_usage="low",
            speed="fast"
        ),
        "BAAI/bge-base-en-v1.5": EmbeddingModelConfig(
            name="BAAI/bge-base-en-v1.5",
            dimensions=768,
            description="Good performance with moderate resource usage",
            performance="very good",
            size="medium",
            recommended_use="Balanced performance and efficiency",
            memory_usage="medium",
            speed="medium"
        ),
        "intfloat/e5-base-v2": EmbeddingModelConfig(
            name="intfloat/e5-base-v2",
            dimensions=768,
            description="Solid performance for general use",
            performance="very good",
            size="medium",
            recommended_use="General applications with good efficiency",
            memory_usage="medium",
            speed="medium"
        )
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[EmbeddingModelConfig]:
        """Get configuration for a specific model"""
        return cls.AVAILABLE_MODELS.get(model_name)
    
    @classmethod
    def get_recommended_model(cls, use_case: str = "production") -> str:
        """Get recommended model based on use case"""
        recommendations = {
            "production": "BAAI/bge-large-en-v1.5",
            "development": "sentence-transformers/all-MiniLM-L12-v2",
            "testing": "all-MiniLM-L6-v2",
            "balanced": "sentence-transformers/all-mpnet-base-v2",
            "multilingual": "intfloat/e5-large-v2",
            "fast": "sentence-transformers/all-MiniLM-L12-v2"
        }
        return recommendations.get(use_case, "BAAI/bge-large-en-v1.5")
    
    @classmethod
    def list_models_by_performance(cls) -> Dict[str, list]:
        """List models grouped by performance level"""
        performance_groups = {
            "excellent": [],
            "very good": [],
            "good": [],
            "fair": []
        }
        
        for model_name, config in cls.AVAILABLE_MODELS.items():
            performance_groups[config.performance].append(model_name)
        
        return performance_groups
    
    @classmethod
    def get_model_comparison(cls) -> str:
        """Get a formatted comparison of all available models"""
        comparison = "\n=== Embedding Model Comparison ===\n\n"
        
        for model_name, config in cls.AVAILABLE_MODELS.items():
            comparison += f"Model: {config.name}\n"
            comparison += f"  Dimensions: {config.dimensions}\n"
            comparison += f"  Performance: {config.performance}\n"
            comparison += f"  Size: {config.size}\n"
            comparison += f"  Memory Usage: {config.memory_usage}\n"
            comparison += f"  Speed: {config.speed}\n"
            comparison += f"  Description: {config.description}\n"
            comparison += f"  Recommended Use: {config.recommended_use}\n"
            comparison += "-" * 50 + "\n"
        
        return comparison
    
    @classmethod
    def validate_model_name(cls, model_name: str) -> bool:
        """Validate if a model name is supported"""
        return model_name in cls.AVAILABLE_MODELS
    
    @classmethod
    def get_model_suggestions(cls, requirements: Dict[str, str]) -> list:
        """Get model suggestions based on requirements"""
        suggestions = []
        
        performance_req = requirements.get("performance", "good")
        memory_req = requirements.get("memory", "medium")
        speed_req = requirements.get("speed", "medium")
        
        for model_name, config in cls.AVAILABLE_MODELS.items():
            score = 0
            
            # Performance scoring
            perf_scores = {"excellent": 4, "very good": 3, "good": 2, "fair": 1}
            if perf_scores.get(config.performance, 0) >= perf_scores.get(performance_req, 2):
                score += 3
            
            # Memory scoring (lower is better for constraints)
            mem_scores = {"low": 3, "medium": 2, "high": 1}
            if mem_scores.get(config.memory_usage, 2) >= mem_scores.get(memory_req, 2):
                score += 2
            
            # Speed scoring
            speed_scores = {"fast": 3, "medium": 2, "slow": 1}
            if speed_scores.get(config.speed, 2) >= speed_scores.get(speed_req, 2):
                score += 1
            
            if score >= 4:  # Threshold for recommendation
                suggestions.append((model_name, score, config))
        
        # Sort by score (descending)
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in suggestions[:3]]  # Top 3 suggestions


def print_model_info(model_name: str) -> None:
    """Print detailed information about a specific model"""
    config = EmbeddingModelManager.get_model_config(model_name)
    if config:
        print(f"\n=== {config.name} ===")
        print(f"Dimensions: {config.dimensions}")
        print(f"Performance: {config.performance}")
        print(f"Size: {config.size}")
        print(f"Memory Usage: {config.memory_usage}")
        print(f"Speed: {config.speed}")
        print(f"Description: {config.description}")
        print(f"Recommended Use: {config.recommended_use}")
    else:
        print(f"Model '{model_name}' not found in available models.")


if __name__ == "__main__":
    # Example usage
    print(EmbeddingModelManager.get_model_comparison())
    
    print("\nRecommendations:")
    print(f"Production: {EmbeddingModelManager.get_recommended_model('production')}")
    print(f"Development: {EmbeddingModelManager.get_recommended_model('development')}")
    print(f"Fast/Testing: {EmbeddingModelManager.get_recommended_model('fast')}")
    
    print("\nModel suggestions for high performance, medium memory:")
    suggestions = EmbeddingModelManager.get_model_suggestions({
        "performance": "excellent",
        "memory": "medium",
        "speed": "medium"
    })
    for suggestion in suggestions:
        print(f"  - {suggestion}")
