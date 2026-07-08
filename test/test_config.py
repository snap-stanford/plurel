import math

import pytest

from plurel.config import Choices


def test_sample_log_uniform_is_log_scaled_within_bounds():
    lo, hi = 1e-4, 0.3
    s = Choices(kind="range", value=[lo, hi]).sample_log_uniform(size=20000)
    assert s.min() >= lo and s.max() <= hi
    # log-uniform puts ~half the mass below the geometric mean (a linear-uniform
    # sampler would put only ~2% there), so this distinguishes the two.
    geo_mean = math.sqrt(lo * hi)
    assert 0.45 < (s < geo_mean).mean() < 0.55


def test_sample_log_uniform_requires_positive_float_range():
    with pytest.raises(ValueError):
        Choices(kind="range", value=[1, 10]).sample_log_uniform()  # int range
    with pytest.raises(ValueError):
        Choices(kind="set", value=[0.1, 0.2]).sample_log_uniform()
