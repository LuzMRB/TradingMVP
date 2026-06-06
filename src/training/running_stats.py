"""
running_stats.py — Normalización online de rewards con Welford's algorithm.
"""

import numpy as np


class RunningMeanStd:
    """
    Calcula media y varianza online (Welford's algorithm).
    Usado para normalizar rewards antes de calcular losses.
    """

    def __init__(self, epsilon: float = 1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).flatten()
        batch_mean  = x.mean()
        batch_var   = x.var()
        batch_count = len(x)

        delta       = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean  += delta * batch_count / total_count
        m_a         = self.var   * self.count
        m_b         = batch_var  * batch_count
        m2          = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var    = m2 / total_count
        self.count  = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip)
