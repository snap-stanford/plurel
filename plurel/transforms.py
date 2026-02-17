import torch

from plurel.config import SCMParams


class MLP:
    def __init__(
        self,
        scm_params: SCMParams,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int = 2,
    ):
        assert num_layers >= 1
        self.scm_params = scm_params
        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        self.weights = [torch.empty(dims[i], dims[i + 1]) for i in range(num_layers)]
        for W in self.weights:
            init_fn = scm_params.initialization_choices.sample_uniform()
            init_fn(W)

    def __call__(self, x):
        for W in self.weights[:-1]:
            act_fn = self.scm_params.activation_choices.sample_uniform()
            x = act_fn(x @ W)
        return x @ self.weights[-1]


class CategoricalEncoder:
    def __init__(
        self,
        scm_params: SCMParams,
        num_embeddings: int,
        embedding_dim: int,
        num_layers: int = 2,
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
            num_layers=num_layers,
        )

    def __call__(self, x: torch.LongTensor):
        return self.mlp(self.E(x))


class CategoricalDecoder:
    def __init__(
        self,
        scm_params: SCMParams,
        num_embeddings: int,
        embedding_dim: int,
        num_layers: int = 2,
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
            num_layers=num_layers,
        )

    def __call__(self, x: torch.Tensor):
        x = self.mlp(x)
        if x.dim() == 1:
            sims = self.E.weight @ x
        else:
            sims = self.E.weight @ x.mT
        return torch.argmax(sims, dim=0)
