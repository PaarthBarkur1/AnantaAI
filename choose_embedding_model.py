#!/usr/bin/env python3
"""
Embedding Model Selection Tool for AnantaAI

This script helps you choose the best embedding model for your specific needs
and hardware constraints.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.embedding_config import EmbeddingModelManager, print_model_info
import torch


def check_system_resources():
    """Check available system resources"""
    print("=== System Resource Check ===")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        gpu_available = True
    else:
        print("❌ No GPU detected - will use CPU")
        gpu_available = False
    
    # Estimate available RAM (simplified)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 System RAM: {ram_gb:.1f} GB")
    except ImportError:
        print("💾 System RAM: Unable to detect (install psutil for details)")
        ram_gb = 8  # Conservative estimate
    
    return gpu_available, ram_gb


def get_user_requirements():
    """Get user requirements through interactive prompts"""
    print("\n=== Requirements Assessment ===")
    
    # Use case
    print("\nWhat is your primary use case?")
    print("1. Production system (highest accuracy needed)")
    print("2. Development/Testing (balance of speed and accuracy)")
    print("3. Resource-constrained environment (speed priority)")
    print("4. Research/Experimentation (flexibility priority)")
    
    use_case_map = {
        "1": "production",
        "2": "development", 
        "3": "fast",
        "4": "balanced"
    }
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice in use_case_map:
            use_case = use_case_map[choice]
            break
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    # Performance requirements
    print("\nWhat performance level do you need?")
    print("1. Excellent (best possible accuracy)")
    print("2. Very Good (high accuracy, good efficiency)")
    print("3. Good (balanced performance)")
    print("4. Fair (basic performance, fastest)")
    
    perf_map = {
        "1": "excellent",
        "2": "very good",
        "3": "good",
        "4": "fair"
    }
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice in perf_map:
            performance = perf_map[choice]
            break
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    # Memory constraints
    print("\nWhat are your memory constraints?")
    print("1. No constraints (can use large models)")
    print("2. Moderate constraints (prefer medium-sized models)")
    print("3. Strict constraints (need small, efficient models)")
    
    memory_map = {
        "1": "high",
        "2": "medium", 
        "3": "low"
    }
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice in memory_map:
            memory = memory_map[choice]
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    return {
        "use_case": use_case,
        "performance": performance,
        "memory": memory,
        "speed": "medium"  # Default
    }


def recommend_models(requirements, gpu_available, ram_gb):
    """Recommend models based on requirements and system resources"""
    print("\n=== Model Recommendations ===")
    
    # Get base recommendations
    suggestions = EmbeddingModelManager.get_model_suggestions(requirements)
    
    # Filter based on system resources
    filtered_suggestions = []
    
    for model_name in suggestions:
        config = EmbeddingModelManager.get_model_config(model_name)
        if not config:
            continue
            
        # Check memory requirements
        if ram_gb < 8 and config.memory_usage == "high":
            continue  # Skip high-memory models on low-RAM systems
        
        if not gpu_available and config.size == "large":
            print(f"⚠️  {model_name}: Large model without GPU may be slow")
        
        filtered_suggestions.append(model_name)
    
    if not filtered_suggestions:
        # Fallback to basic model
        filtered_suggestions = ["sentence-transformers/all-MiniLM-L12-v2"]
        print("⚠️  Using fallback recommendation due to system constraints")
    
    # Display recommendations
    print(f"\n🎯 Top recommendations for your setup:")
    for i, model_name in enumerate(filtered_suggestions[:3], 1):
        config = EmbeddingModelManager.get_model_config(model_name)
        print(f"\n{i}. {model_name}")
        print(f"   📊 Performance: {config.performance}")
        print(f"   💾 Memory Usage: {config.memory_usage}")
        print(f"   ⚡ Speed: {config.speed}")
        print(f"   📝 {config.description}")
    
    return filtered_suggestions[0] if filtered_suggestions else "sentence-transformers/all-MiniLM-L12-v2"


def update_config_files(recommended_model):
    """Update configuration files with the recommended model"""
    print(f"\n=== Updating Configuration ===")
    
    files_to_update = [
        "backend/qna.py",
        "backend/main.py", 
        "app.py",
        "test_system.py"
    ]
    
    print(f"Recommended model: {recommended_model}")
    print("\nTo update your configuration:")
    print(f"1. Set embedding_model = '{recommended_model}' in QAConfig")
    print("2. Or update the model_name variables in the following files:")
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            print(f"   - {file_path}")
    
    print(f"\n💡 You can also set the model in your QAConfig:")
    print(f"   config = QAConfig(embedding_model='{recommended_model}')")


def main():
    """Main function"""
    print("🤖 AnantaAI Embedding Model Selection Tool")
    print("=" * 50)
    
    # Check system resources
    gpu_available, ram_gb = check_system_resources()
    
    # Get user requirements
    requirements = get_user_requirements()
    
    # Get recommendations
    recommended_model = recommend_models(requirements, gpu_available, ram_gb)
    
    # Show detailed info about recommended model
    print(f"\n=== Detailed Information for Recommended Model ===")
    print_model_info(recommended_model)
    
    # Show how to update configuration
    update_config_files(recommended_model)
    
    # Show all available models
    print(f"\n=== All Available Models ===")
    print("Run the following to see all models:")
    print("python -c \"from backend.embedding_config import EmbeddingModelManager; print(EmbeddingModelManager.get_model_comparison())\"")
    
    print(f"\n✅ Model selection complete!")
    print(f"🎯 Recommended: {recommended_model}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Model selection cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure you're running this from the AnantaAI root directory.")
