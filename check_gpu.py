#!/usr/bin/env python3
"""
GPU Detection and Information Script for AnantaAI
"""

import sys

def check_gpu_availability():
    """Check GPU availability and provide detailed information"""
    print("AnantaAI GPU Detection")
    print("=" * 40)
    
    # Check PyTorch installation
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
    except ImportError:
        print("[FAIL] PyTorch not installed")
        print("Install with: pip install torch")
        return False
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.version.cuda}")
        
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        print(f"[OK] Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"[OK] GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Check GPU memory usage
            if torch.cuda.is_available():
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"     Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")
        
        # Test GPU functionality
        try:
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.matmul(test_tensor, test_tensor)
            print("[OK] GPU computation test passed")
            return True
        except Exception as e:
            print(f"[WARN] GPU computation test failed: {e}")
            return False
            
    else:
        print("[INFO] CUDA not available - will use CPU")
        
        # Check if CUDA is installed but not working
        try:
            print(f"[INFO] CUDA version: {torch.version.cuda}")
            print("[WARN] CUDA is installed but not accessible")
            print("       This might be due to:")
            print("       - Incompatible GPU drivers")
            print("       - Wrong PyTorch version")
            print("       - CUDA version mismatch")
        except:
            print("[INFO] CUDA not installed")
            print("       Install CUDA-enabled PyTorch with:")
            print("       pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        return False

def check_model_requirements():
    """Check if the system can handle the models"""
    print("\nModel Requirements Check")
    print("=" * 40)
    
    try:
        import torch
        
        # Check available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[INFO] GPU Memory: {gpu_memory:.1f}GB")
            
            if gpu_memory >= 4.0:
                print("[OK] Sufficient GPU memory for Qwen2.5-0.5B")
            elif gpu_memory >= 2.0:
                print("[WARN] Limited GPU memory - may need to use CPU fallback")
            else:
                print("[WARN] Very limited GPU memory - CPU recommended")
        else:
            print("[INFO] Will use CPU - ensure sufficient RAM (4GB+ recommended)")
            
    except Exception as e:
        print(f"[ERROR] Could not check requirements: {e}")

def main():
    """Main function"""
    gpu_available = check_gpu_availability()
    check_model_requirements()
    
    print("\n" + "=" * 40)
    if gpu_available:
        print("SUCCESS: GPU is available and working!")
        print("AnantaAI will use GPU acceleration for faster inference.")
    else:
        print("INFO: GPU not available - will use CPU")
        print("AnantaAI will work but inference will be slower.")
    
    print("\nTo start AnantaAI:")
    print("1. Run: python setup_venv.py (if not done)")
    print("2. Run: start.bat (Windows) or ./start.sh (Linux/Mac)")
    
    return 0 if gpu_available else 1

if __name__ == "__main__":
    sys.exit(main())
