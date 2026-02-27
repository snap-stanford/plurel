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
        ar_rho: float = 0.0,
    ):
        assert cycle_frequency <= num_points
        assert 0.0 <= ar_rho < 1.0
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
        self.ar_rho = ar_rho
        self._noise_state = 0.0

    def _get_noise_val(self):
        self._noise_state = self.ar_rho * self._noise_state + np.random.randn() * self.noise_scale
        return min(max(self._noise_state, self.min_value), self.max_value)

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


class UniformSourceGenerator:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def get_value(self, row_idx: int) -> float:
        return float(np.random.uniform(self.low, self.high))


class GaussianSourceGenerator:
    def __init__(self, mean: float, std: float, low: float, high: float):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high

    def get_value(self, row_idx: int) -> float:
        return float(np.clip(np.random.normal(self.mean, self.std), self.low, self.high))


class BetaSourceGenerator:
    def __init__(self, alpha: float, beta: float, scale: float, offset: float):
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.offset = offset

    def get_value(self, row_idx: int) -> float:
        return float(np.random.beta(self.alpha, self.beta) * self.scale + self.offset)


class MixedSourceGenerator:
    def __init__(self, generators: list):
        self.generators = generators

    def get_value(self, row_idx: int) -> float:
        gen = self.generators[np.random.randint(len(self.generators))]
        return gen.get_value(row_idx)


class IIDCategoricalGenerator:
    def __init__(self, num_categories: int):
        self.probs = np.random.dirichlet(np.ones(num_categories))
        self.num_categories = num_categories

    def get_value(self, row_idx: int) -> int:
        return int(np.random.choice(self.num_categories, p=self.probs))
