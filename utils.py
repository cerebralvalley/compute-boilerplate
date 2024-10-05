import torch
import psutil
from transformers import AutoModelForCausalLM
import shutil
from huggingface_hub import model_info, login
from config import Config
import json
from enum import Enum
from pydantic import BaseModel

def serialize(request: BaseModel) -> str:
    """
    Turns a pydantic model (request param) into serialized JSON string format.
    Uses the EnumEncoder class to get the value of the Enum so we can model dump without error.
    """
    class EnumEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Enum):
                return obj.value
            return super().default(obj)
    return json.dumps(request.model_dump(), cls=EnumEncoder)

def estimate_model_size(model_name):
    """
    Estimates the model size with the model's name, considering different precisions.
    Uses the model's configuration file to get the number of parameters.
    """    
    if Config.HUGGINGFACE_ACCESS_TOKEN:
        login(token=Config.HUGGINGFACE_ACCESS_TOKEN)

    info = model_info(model_name)
    tensor_params = info.safetensors.parameters
    total_bytes = 0
    for key, value in tensor_params.items():
        if '32' in key:
            total_bytes += value * 4  # 32-bit float
        elif '16' in key:
            total_bytes += value * 2  # 16-bit float
        elif '8' in key:
            total_bytes += value  # 8-bit integer
        elif '4' in key:
            total_bytes += value / 2  # 4-bit integer
        else:
            total_bytes += value * 4  # default to 32-bit float if precision is not specified
    estimated_size_gb = total_bytes / (1024 ** 3)
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
    Actually calculates the size of the model after the estimate check by looping through each downloaded parameter.
    Raises an error if the model won't fit in memory or if there's not enough disk space.
    """
    print("\n" + "=" * 50)
    print(f"System Resource Check for Model: {model_name}")
    print("=" * 50)

    estimated_size = estimate_model_size(model_name)  # estimate the model's size before actually downloading it
    available_disk_space = get_available_disk_space()
    available_vram = get_available_vram()
    available_memory = get_available_memory()

    if estimated_size > available_disk_space:
        raise ValueError(f"Estimated model size ({estimated_size:.2f} GB) exceeds available disk space ({available_disk_space:.2f} GB). Refusing to download model.")
    
    if available_vram > 0 and estimated_size > available_vram:
        raise ValueError(f"Estimated model size ({estimated_size:.2f} GB) exceeds available GPU memory ({available_vram:.2f} GB). Refusing to download model.")
    elif available_vram == 0 and estimated_size > available_memory:
        raise ValueError(f"Estimated model size ({estimated_size:.2f} GB) exceeds available system memory ({available_memory:.2f} GB). Refusing to download model.")
    
    print("System requirements estimate check passed. Proceeding to ensure model feasibility...")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_size = calculate_model_size(model)
    
    del model
    torch.cuda.empty_cache()

    print(f"Actual model size: {model_size:.2f} GB")

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
