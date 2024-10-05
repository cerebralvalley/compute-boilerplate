import torch
import psutil
from transformers import AutoModelForCausalLM
from config import Config

def estimate_model_size(model) -> float:
    """
    Estimates the size of the model, in GB.
    """
    total_size = 0
    for param in model.parameters():
        if param.dtype == torch.float32:
            bits = 32
        elif param.dtype == torch.float16 or param.dtype == torch.bfloat16:
            bits = 16
        elif param.dtype == torch.int8:
            bits = 8
        else:
            print("dtype not detected. Falling back to quanitzation bits config...")
            bits = Config.QUANTIZATION_BITS  # fallback to config

        total_size += param.numel() * (bits / 8)
    
    return total_size / (1024 ** 3)

def check_system_resources(model_name):
    """
    Checks the model size against system parameter, with or without GPU. 
    """
    print("\n" + "=" * 50)
    print(f"System Resource Check for Model: {model_name}")
    print("=" * 50)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_size = estimate_model_size(model)
    
    del model
    torch.cuda.empty_cache()

    print(f"Actual model size with {Config.QUANTIZATION_BITS}-bit precision: {model_size:.2f} GB")

    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        available_memory -= (torch.cuda.memory_reserved(0) + torch.cuda.memory_allocated(0)) / (1024 ** 3)
        
        print("\nGPU Information:")
        print(f"  - Available GPU memory: {available_memory:.2f} GB")
        
        if model_size > available_memory:
            print("  - WARNING: The model may not fit in GPU memory!")
        else:
            print("  - The model should fit in GPU memory.")
    else:
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        
        print("\nCPU Information:")
        print("  - CUDA is not available. Using CPU.")
        print(f"  - Available system memory: {available_memory:.2f} GB")
        
        if model_size > available_memory:
            print("  - WARNING: The model may not fit in system memory!")
        else:
            print("  - The model SHOULD fit in system memory.")

    print("\nMemory Requirement:")
    print(f"  - Required: {model_size:.2f} GB")
    print(f"  - Available: {available_memory:.2f} GB")
    print("=" * 50 + "\n")
