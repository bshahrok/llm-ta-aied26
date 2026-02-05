"""
Memory and GPU management utilities.
"""

import gc
import logging
import os
from typing import Dict

import torch

logger = logging.getLogger(__name__)


def clear_cache():
    """Clears GPU cache and runs garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_memory_info() -> Dict[str, float]:
    """Returns current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    free = total - allocated

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "free_gb": free
    }


def log_memory_usage():
    """Logs current memory usage."""
    info = get_memory_info()
    if info:
        logger.debug(
            f"GPU Memory - Allocated: {info['allocated_gb']:.2f}GB, "
            f"Free: {info['free_gb']:.2f}GB, Total: {info['total_gb']:.2f}GB"
        )


def setup_memory_optimization():
    """Sets up environment variables for better memory management."""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'