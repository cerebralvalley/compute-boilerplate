import torch
import psutil
from transformers import AutoModelForCausalLM
import shutil
from huggingface_hub import model_info

def estimate_model_size(model_name):
    """
    Estimates the model size with the model's name, assuming full (32 bit) precision.
    Uses the model's configuration file to get the number of parameters.
    """    
    info = model_info(model_name)
    tensor_params = info.safetensors.parameters
    num_parameters = sum(tensor_params.values())  # sum all values in the dictionary, regardless of precision
    estimated_size_bytes = num_parameters * 4  # assume 4 bytes per parameter (32-bit float)
    estimated_size_gb = estimated_size_bytes / (1024 ** 3)
    return estimated_size_gb

def calculate_model_size(model) -> float:
    """
    Calculates the size of the model, in GB, counting param by param.
    """
    total_size = 0
    for param in model.parameters():
        if param.dtype == torch.float32:
            bits = 32
        elif param.dtype == torch.float16 or param.dtype == torch.bfloat16:
            bits = 16
        elif param.dtype == torch.int8:
            bits = 8
        elif param.dtype == torch.int4:
            bits = 4
        else:
            print("dtype not detected. Falling back to full precision...")
            bits = 32

        total_size += param.numel() * (bits / 8)
    
    return total_size / (1024 ** 3)

def get_available_vram():
    """
    Returns available VRAM in GB if CUDA is available, otherwise returns 0.
    """
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        available_memory -= (torch.cuda.memory_reserved(0) + torch.cuda.memory_allocated(0)) / (1024 ** 3)
        return available_memory
    return 0

def get_available_memory():
    """
    Returns available system memory in GB.
    """
    return psutil.virtual_memory().available / (1024 ** 3)

def get_available_disk_space():
    """
    Returns available disk space in GB.
    """
    _, _, free = shutil.disk_usage("/")
    return free // (2**30)

def check_system_resources(model_name):
    """
    Checks the model size against system parameters, with or without GPU, and disk space.
    Raises an error if the model won't fit in memory or if there's not enough disk space.
    """
    print("\n" + "=" * 50)
    print(f"System Resource Check for Model: {model_name}")
    print("=" * 50)

    estimated_size = estimate_model_size(model_name)
    available_disk_space = get_available_disk_space()

    if estimated_size > available_disk_space:
        raise ValueError(f"Estimated model size ({estimated_size:.2f} GB) exceeds available disk space ({available_disk_space:.2f} GB). Refusing to download model.")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_size = calculate_model_size(model)
    
    del model
    torch.cuda.empty_cache()

    print(f"Actual model size: {model_size:.2f} GB")

    available_vram = get_available_vram()
    available_memory = get_available_memory()

    if available_vram > 0:
        print("\nGPU Information:")
        print(f"  - Available GPU memory: {available_vram:.2f} GB")
        
        if model_size > available_vram:
            raise ValueError(f"The model size ({model_size:.2f} GB) exceeds available GPU memory ({available_vram:.2f} GB).")
        else:
            print("  - The model should fit in GPU memory.")
    else:
        print("\nCPU Information:")
        print("  - CUDA is not available. Using CPU.")
        print(f"  - Available system memory: {available_memory:.2f} GB")
        
        if model_size > available_memory:
            raise ValueError(f"The model size ({model_size:.2f} GB) exceeds available system memory ({available_memory:.2f} GB).")
        else:
            print("  - The model should fit in system memory.")

    print("\nMemory Requirement:")
    print(f"  - Required: {model_size:.2f} GB")
    print(f"  - Available: {max(available_vram, available_memory):.2f} GB")

    print("\nDisk Space:")
    print(f"  - Available: {available_disk_space:.2f} GB")
    print(f"  - Required: {model_size:.2f} GB")

    if available_disk_space < model_size:
        raise ValueError(f"Not enough disk space to store the model. Required: {model_size:.2f} GB, Available: {available_disk_space:.2f} GB")
    else:
        print("  - Sufficient disk space available.")

    print("=" * 50 + "\n")
