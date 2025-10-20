# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 21:58:03 2025

@author: Ruining
"""

# utils/log.py
import os
import time
import numpy as np

class Logger:
    """Simple logger for CPPO training process."""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.start_time = time.time()

    def log(self, episode: int, reward: float, cost: float, constraint_violation: float = 0.0):
        """Print and record episode statistics."""
        elapsed = time.time() - self.start_time
        print(f"[{episode:05d}] Reward: {reward:.2f} | Cost: {cost:.2f} | Constraint: {constraint_violation:.3f} | Time: {elapsed:.1f}s")

    def save(self, filename: str, data: dict):
        """Save summary data to a .npy file."""
        np.save(os.path.join(self.log_dir, filename), data)
