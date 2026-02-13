"""
Synthetic time-series data
"""

import math

import numpy as np


class Cycle:
    def __init__(self, min_value: float, max_value: float, frequency: int, scale: float):
        self.min_value = min_value
        self.max_value = max_value
        self.frequency = frequency
        self.scale = scale

    def get_value(self, row_idx: int) -> float:
        x = (row_idx / self.frequency) * math.pi
        val = self.scale * math.sin(x)
        return min(max(val, self.min_value), self.max_value)


class Trend:
    def __init__(
        self,
        num_points: int,
        min_value: float,
        max_value: float,
        alpha: float,
        scale: float,
    ):
        self.num_points = num_points
        self.min_value = min_value
        self.max_value = max_value
        self.alpha = alpha
        self.scale = scale

    def get_value(self, row_idx: int) -> float:
        x = row_idx / self.num_points
        value = self.scale * math.pow(x, self.alpha) + self.min_value
        return min(value, self.max_value)


class TSDataGenerator:
    def __init__(
        self,
        num_points: int,
        min_value: float,
        max_value: float,
        trend_alpha: float,
        trend_scale: float,
        cycle_frequency: float,
        cycle_scale: float,
        noise_scale: float = 0.05,
    ):
        assert cycle_frequency <= num_points
        self.num_points = num_points
        self.min_value = min_value
        self.max_value = max_value
        self.trend = Trend(
            num_points=num_points,
            min_value=min_value,
            max_value=max_value,
            alpha=trend_alpha,
            scale=trend_scale,
        )
        self.cycle = Cycle(
            min_value=min_value,
            max_value=max_value,
            frequency=cycle_frequency,
            scale=cycle_scale,
        )
        self.noise_scale = noise_scale

    def _get_noise_val(self):
        noise_val = np.random.randn()
        noise_val *= self.noise_scale
        return min(max(noise_val, self.min_value), self.max_value)

    def get_value(self, row_idx):
        trend_val = self.trend.get_value(row_idx=row_idx)
        cycle_val = self.cycle.get_value(row_idx=row_idx)
        noise_val = self._get_noise_val()
        return (trend_val + cycle_val + noise_val) / 3


class CategoricalTSDataGenerator:
    def __init__(self, ts_data_gens: list[TSDataGenerator]):
        self.ts_data_gens = ts_data_gens

    def get_value(self, row_idx):
        vals = [ts_data_gen.get_value(row_idx=row_idx) for ts_data_gen in self.ts_data_gens]
        vals = np.array(vals)
        exp_vals = np.exp(vals - np.max(vals))  # stable exponent
        probs = exp_vals / exp_vals.sum()
        category_idx = np.random.choice(len(vals), p=probs)
        return int(category_idx)
