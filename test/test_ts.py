import numpy as np
import pytest

from plurel.ts import (
    BetaSourceGenerator,
    CategoricalTSDataGenerator,
    Cycle,
    GaussianSourceGenerator,
    IIDCategoricalGenerator,
    MixedSourceGenerator,
    Trend,
    TSDataGenerator,
    UniformSourceGenerator,
)

N = 1000


def _make_ts_gen(num_points=N, ar_rho=0.0):
    return TSDataGenerator(
        num_points=num_points,
        min_value=-1.0,
        max_value=1.0,
        trend_alpha=1.0,
        trend_scale=1.0,
        cycle_frequency=num_points // 4,
        cycle_scale=0.5,
        noise_scale=0.05,
        ar_rho=ar_rho,
    )


def test_cycle_finite_and_bounded():
    cycle = Cycle(min_value=-1.0, max_value=1.0, frequency=100, scale=2.0)
    values = [cycle.get_value(row_idx=i) for i in range(N)]
    assert all(np.isfinite(v) for v in values)
    assert all(-1.0 <= v <= 1.0 for v in values)


def test_cycle_negative_scale():
    cycle = Cycle(min_value=-1.0, max_value=1.0, frequency=50, scale=-2.0)
    values = [cycle.get_value(row_idx=i) for i in range(N)]
    assert all(np.isfinite(v) for v in values)
    assert all(-1.0 <= v <= 1.0 for v in values)


def test_trend_finite_and_bounded():
    trend = Trend(num_points=N, min_value=0.0, max_value=5.0, alpha=1.5, scale=10.0)
    values = [trend.get_value(row_idx=i) for i in range(N)]
    assert all(np.isfinite(v) for v in values)
    assert all(v <= 5.0 for v in values)


def test_trend_zero_alpha():
    trend = Trend(num_points=N, min_value=-1.0, max_value=1.0, alpha=0.0, scale=1.0)
    values = [trend.get_value(row_idx=i) for i in range(N)]
    assert all(np.isfinite(v) for v in values)


@pytest.mark.parametrize("ar_rho", [0.0, 0.5, 0.9])
def test_ts_data_generator_finite(ar_rho):
    np.random.seed(0)
    gen = _make_ts_gen(ar_rho=ar_rho)
    values = [gen.get_value(row_idx=i) for i in range(N)]
    assert all(np.isfinite(v) for v in values)


def test_ts_data_generator_correct_length():
    np.random.seed(0)
    gen = _make_ts_gen(num_points=200)
    values = [gen.get_value(row_idx=i) for i in range(200)]
    assert len(values) == 200


def test_categorical_ts_data_generator_valid_range():
    np.random.seed(0)
    num_categories = 4
    gen = CategoricalTSDataGenerator(ts_data_gens=[_make_ts_gen() for _ in range(num_categories)])
    values = [gen.get_value(row_idx=i) for i in range(N)]
    assert all(isinstance(v, int) for v in values)
    assert all(0 <= v < num_categories for v in values)


def test_categorical_ts_data_generator_all_categories_reachable():
    np.random.seed(0)
    num_categories = 3
    gen = CategoricalTSDataGenerator(ts_data_gens=[_make_ts_gen() for _ in range(num_categories)])
    values = {gen.get_value(row_idx=i) for i in range(N)}
    assert values == set(range(num_categories))


def test_uniform_source_generator_finite():
    gen = UniformSourceGenerator(low=-100.0, high=100.0)
    assert all(np.isfinite(gen.get_value(row_idx=i)) for i in range(N))


def test_uniform_source_generator_bounded():
    gen = UniformSourceGenerator(low=0.0, high=1.0)
    values = [gen.get_value(row_idx=i) for i in range(N)]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_gaussian_source_generator_finite():
    gen = GaussianSourceGenerator(mean=0.0, std=10.0, low=-100.0, high=100.0)
    assert all(np.isfinite(gen.get_value(row_idx=i)) for i in range(N))


def test_gaussian_source_generator_clipped():
    gen = GaussianSourceGenerator(mean=0.0, std=1e6, low=-5.0, high=5.0)
    values = [gen.get_value(row_idx=i) for i in range(N)]
    assert all(-5.0 <= v <= 5.0 for v in values)


def test_beta_source_generator_finite():
    gen = BetaSourceGenerator(alpha=0.5, beta=0.5, scale=200.0, offset=-100.0)
    assert all(np.isfinite(gen.get_value(row_idx=i)) for i in range(N))


def test_beta_source_generator_bounded():
    gen = BetaSourceGenerator(alpha=2.0, beta=2.0, scale=1.0, offset=0.0)
    values = [gen.get_value(row_idx=i) for i in range(N)]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_mixed_source_generator_finite():
    np.random.seed(0)
    gen = MixedSourceGenerator(
        generators=[
            UniformSourceGenerator(low=-10.0, high=10.0),
            GaussianSourceGenerator(mean=0.0, std=5.0, low=-20.0, high=20.0),
            BetaSourceGenerator(alpha=0.5, beta=2.0, scale=10.0, offset=0.0),
        ]
    )
    assert all(np.isfinite(gen.get_value(row_idx=i)) for i in range(N))


def test_iid_categorical_generator_valid():
    np.random.seed(0)
    gen = IIDCategoricalGenerator(num_categories=5)
    values = [gen.get_value(row_idx=i) for i in range(N)]
    assert all(0 <= v < 5 for v in values)


def test_iid_categorical_generator_all_categories_reachable():
    np.random.seed(0)
    gen = IIDCategoricalGenerator(num_categories=3)
    values = {gen.get_value(row_idx=i) for i in range(N)}
    assert values == {0, 1, 2}
