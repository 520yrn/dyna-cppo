# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 21:57:49 2025

@author: Ruining
"""

import random
import numpy as np
import torch

class LinearSchedule:
    """Linearly decay a value from start to end over given total steps."""

    def __init__(self, start: float, end: float, total_steps: int):
        self.start = start
        self.end = end
        self.total_steps = total_steps

    def value(self, step: int) -> float:
        """Return current value at given step."""
        fraction = min(float(step) / self.total_steps, 1.0)
        return self.start + fraction * (self.end - self.start)
