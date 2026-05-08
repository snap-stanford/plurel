import math

import numpy as np
import torch

from plurel.config import RandomFunctionActivation, SCMParams


class MLP:
    def __init__(
        self,
        scm_params: SCMParams,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
    ):
        num_layers = int(scm_params.mlp_num_layers_choices.sample_uniform())
        assert num_layers >= 1
        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        self.weights = [torch.empty(dims[i], dims[i + 1]) for i in range(num_layers)]
        for W in self.weights:
            init_fn = scm_params.initialization_choices.sample_uniform()
            init_fn(W)
            # Bernoulli sparsity mask with variance compensation; density=1.0 => identity
            density = float(scm_params.mlp_weight_density_choices.sample_uniform())
            mask = torch.bernoulli(torch.full_like(W, density))
            W.data *= mask / max(density, 1e-6)
        self.act_scales = []
        for _ in range(num_layers - 1):
            act_fn = scm_params.activation_choices.sample_uniform()
            if isinstance(act_fn, RandomFunctionActivation):
                # The instance in activation_choices is a marker; instantiate
                # a fresh RFA so this layer's activation has its own frozen
                # params, independent of any other layer that picked RFA.
                act_fn = RandomFunctionActivation()
            self.act_scales.append((act_fn, math.exp(np.random.uniform(-1.0, 1.0))))

    def __call__(self, x):
        for W, (act_fn, scale) in zip(self.weights[:-1], self.act_scales):
            x = torch.clamp(scale * act_fn(x @ W), -1e6, 1e6)
        return x @ self.weights[-1]


class CategoricalEncoder:
    def __init__(
        self,
        scm_params: SCMParams,
        num_embeddings: int,
        embedding_dim: int,
    ):
        self.E = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, _freeze=True
        )
        init_fn = scm_params.initialization_choices.sample_uniform()
        init_fn(self.E.weight)
        self.mlp = MLP(
            scm_params=scm_params,
            in_dim=embedding_dim,
            hid_dim=embedding_dim,
            out_dim=embedding_dim,
        )

    def __call__(self, x: torch.LongTensor):
        return self.mlp(self.E(x))


class CategoricalDecoder:
    def __init__(
        self,
        scm_params: SCMParams,
        num_embeddings: int,
        embedding_dim: int,
    ):
        self.E = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, _freeze=True
        )
        init_fn = scm_params.initialization_choices.sample_uniform()
        init_fn(self.E.weight)
        self.mlp = MLP(
            scm_params=scm_params,
            in_dim=embedding_dim,
            hid_dim=embedding_dim,
            out_dim=embedding_dim,
        )

    def __call__(self, x: torch.Tensor):
        x = self.mlp(x)
        if x.dim() == 1:
            sims = self.E.weight @ x
        else:
            sims = self.E.weight @ x.mT
        return torch.argmax(sims, dim=0)
