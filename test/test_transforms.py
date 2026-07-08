import pytest
import torch

from plurel.config import Choices, SCMParams
from plurel.transforms import MLPMechanism, TreeMechanism, make_mechanism
from plurel.utils import set_random_seed

# (in_dim, out_dim) for the three sites make_mechanism is used at:
# numerical encoder (1 -> emb), numerical decoder (emb -> 1), categorical (emb -> emb).
SHAPE_SITES = [(1, 32), (32, 1), (32, 32)]
TREE_MODELS = SCMParams().tree_model_choices.value


def _params(**overrides):
    return SCMParams(**overrides)


def test_make_mechanism_dispatch():
    mlp = make_mechanism(
        _params(mechanism_type_choices=Choices(kind="set", value=["mlp"])), 32, 32, 1
    )
    tree = make_mechanism(
        _params(mechanism_type_choices=Choices(kind="set", value=["tree"])), 32, 32, 1
    )
    assert isinstance(mlp, MLPMechanism)
    assert isinstance(tree, TreeMechanism)


@pytest.mark.parametrize("tree_model", TREE_MODELS)
def test_tree_mechanism_shape_and_finite(tree_model):
    p = _params(tree_model_choices=Choices(kind="set", value=[tree_model]))
    for in_dim, out_dim in SHAPE_SITES:
        y = TreeMechanism(scm_params=p, in_dim=in_dim, hid_dim=32, out_dim=out_dim)(
            torch.randn(64, in_dim)
        )
        assert y.shape == (64, out_dim)
        assert torch.isfinite(y).all()


def test_tree_mechanism_fits_once_and_reuses():
    m = TreeMechanism(scm_params=_params(), in_dim=8, hid_dim=32, out_dim=4)
    m(torch.randn(50, 8))
    fitted_model = m.model
    y = m(torch.randn(23, 8))
    assert m.model is fitted_model  # a later, differently-sized call must not refit
    assert y.shape == (23, 4)


def test_tree_mechanism_reproducible_under_seed():
    x = torch.randn(40, 16)
    outs = []
    for _ in range(2):
        set_random_seed(123)
        outs.append(TreeMechanism(scm_params=_params(), in_dim=16, hid_dim=32, out_dim=4)(x))
    assert torch.equal(outs[0], outs[1])
