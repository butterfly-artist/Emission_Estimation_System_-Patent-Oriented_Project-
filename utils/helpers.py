import logging
import time
import numpy as np
from typing import Any

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the specified timestamp format:
    [YYYY-MM-DD HH:MM:SS] INFO modulename: message
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        if name in ['weights', 'adaptive', 'residuals', 'inversion']:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False
    return logger

_logger = get_logger("helpers")

def normalize_minmax(arr: Any) -> np.ndarray:
    """Scales array to [0,1]."""
    arr = np.asarray(arr, dtype=float)
    a_min = np.min(arr)
    a_max = np.max(arr)
    if a_max == a_min:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min)

def normalize_zscore(arr: Any) -> np.ndarray:
    """Scales array to zero mean unit variance."""
    arr = np.asarray(arr, dtype=float)
    a_mean = np.mean(arr)
    a_std = np.std(arr)
    if a_std == 0:
        return np.zeros_like(arr)
    return (arr - a_mean) / a_std

def array_summary(name: str, arr: Any) -> None:
    """Prints a one-line summary of an array."""
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f"{name} | {arr.shape} | empty array")
    else:
        print(f"{name} | {arr.shape} | {np.min(arr):.4f} | {np.max(arr):.4f} | {np.mean(arr):.4f} | {np.std(arr):.4f}")

class Timer:
    """Context manager that prints elapsed time."""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.3f}s")

def save_array(path: str, arr: Any) -> None:
    """Saves numpy .npy file with logged confirmation."""
    arr = np.asarray(arr)
    np.save(path, arr)
    _logger.info(f"Saved array to {path}")

def load_array(path: str) -> np.ndarray:
    """Loads .npy file with logged confirmation and shape print."""
    arr = np.load(path)
    _logger.info(f"Loaded array from {path}")
    print(f"Shape: {arr.shape}")
    return arr

if __name__ == "__main__":
    import os
    test_logger = get_logger("self_test")
    test_logger.info("Starting helpers.py self-test.")
    
    # Create a small test array
    test_arr = np.array([[-10, 0, 10], [20, 30, 40]])
    array_summary("Original", test_arr)
    
    with Timer("minmax norm  "):
        norm_arr = normalize_minmax(test_arr)
    array_summary("MinMax", norm_arr)
    
    with Timer("zscore norm  "):
        z_arr = normalize_zscore(test_arr)
    array_summary("ZScore", z_arr)
    
    test_path = "test_array.npy"
    save_array(test_path, norm_arr)
    
    with Timer("load         "):
        loaded = load_array(test_path)
    
    if os.path.exists(test_path):
        os.remove(test_path)
        
    test_logger.info("Self-test completed.")
