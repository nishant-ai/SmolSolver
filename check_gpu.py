#!/usr/bin/env python3
"""
GPU Detection and Information Script
Checks all available GPUs and their specifications
"""

import sys

print("="*70)
print("GPU DETECTION AND INFORMATION")
print("="*70)

# ============================================================================
# 1. CHECK NVIDIA-SMI
# ============================================================================
print("\n📊 NVIDIA-SMI OUTPUT:")
print("-"*70)

import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
except FileNotFoundError:
    print("❌ nvidia-smi not found. CUDA drivers may not be installed.")
except Exception as e:
    print(f"❌ Error running nvidia-smi: {e}")

# ============================================================================
# 2. CHECK PYTORCH CUDA
# ============================================================================
print("\n🔥 PYTORCH CUDA INFORMATION:")
print("-"*70)

try:
    import torch
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        
        print("\n📋 GPU Details:")
        print("-"*70)
        
        for i in range(torch.cuda.device_count()):
            print(f"\n🎮 GPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            
            # Get memory info
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-Processor Count: {props.multi_processor_count}")
            
            # Current memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"  Memory Allocated: {allocated:.2f} GB")
                print(f"  Memory Reserved: {reserved:.2f} GB")
                print(f"  Memory Free: {total_memory - reserved:.2f} GB")
    else:
        print("❌ CUDA is not available in PyTorch")
        print("\nPossible reasons:")
        print("  1. No NVIDIA GPU detected")
        print("  2. CUDA drivers not installed")
        print("  3. PyTorch installed without CUDA support")
        print("\nTo install PyTorch with CUDA:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
except ImportError:
    print("❌ PyTorch not installed")
    print("Install with: pip install torch")
except Exception as e:
    print(f"❌ Error checking PyTorch CUDA: {e}")

# ============================================================================
# 3. CHECK TENSORFLOW (if available)
# ============================================================================
print("\n🤖 TENSORFLOW GPU INFORMATION:")
print("-"*70)

try:
    import tensorflow as tf
    
    print(f"TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"Number of GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    else:
        print("❌ No GPUs detected by TensorFlow")
        
except ImportError:
    print("ℹ️  TensorFlow not installed (optional)")
except Exception as e:
    print(f"⚠️  Error checking TensorFlow: {e}")

# ============================================================================
# 4. QUICK CUDA TEST
# ============================================================================
print("\n🧪 QUICK CUDA TEST:")
print("-"*70)

try:
    import torch
    
    if torch.cuda.is_available():
        # Create a small tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        
        print("✅ CUDA test successful! GPU computation working.")
        print(f"   Test tensor shape: {z.shape}")
        print(f"   Test tensor device: {z.device}")
    else:
        print("⚠️  Cannot run CUDA test - no GPU available")
        
except Exception as e:
    print(f"❌ CUDA test failed: {e}")

# ============================================================================
# 5. SYSTEM INFORMATION
# ============================================================================
print("\n💻 SYSTEM INFORMATION:")
print("-"*70)

import platform
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Architecture: {platform.machine()}")

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
print("\n💡 RECOMMENDATIONS:")
print("-"*70)

try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                          for i in range(gpu_count)) / (1024**3)
        
        print(f"\n✅ You have {gpu_count} GPU(s) with {total_memory:.1f} GB total memory")
        
        if gpu_count == 1:
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if mem >= 80:
                print("   → Perfect for training! (A100 80GB detected)")
                print("   → Use batch_size=16, gradient_accumulation=2")
            elif mem >= 40:
                print("   → Great for training! (A100 40GB detected)")
                print("   → Use batch_size=16, gradient_accumulation=2")
            elif mem >= 24:
                print("   → Good for training! (RTX 3090/4090 or A5000)")
                print("   → Use batch_size=12, gradient_accumulation=2")
            elif mem >= 16:
                print("   → Suitable for training! (V100 or RTX A4000)")
                print("   → Use batch_size=10, gradient_accumulation=2")
            elif mem >= 12:
                print("   → Can train with smaller batches (T4 or RTX 3080)")
                print("   → Use batch_size=8, gradient_accumulation=2")
            else:
                print("   → Limited memory - reduce batch size")
                print("   → Use batch_size=4, gradient_accumulation=4")
                
        elif gpu_count >= 2:
            print(f"   → Multi-GPU setup detected!")
            print(f"   → Training will automatically distribute across GPUs")
            print(f"   → Use batch_size=8-16 per device")
    else:
        print("❌ No GPU detected")
        print("   → Training will be very slow on CPU")
        print("   → Consider using Google Colab, Kaggle, or cloud GPU instances")
        
except:
    pass

print("\n" + "="*70)
print("✅ GPU CHECK COMPLETE")
print("="*70 + "\n")
