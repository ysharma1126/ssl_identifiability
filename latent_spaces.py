from typing import Callable, List
from spaces import Space
import torch


class LatentSpace:
    """Combines a topological space with a marginal and conditional density to sample from."""

    def __init__(
        self, space: Space, sample_marginal: Callable, sample_conditional: Callable
    ):
        self.space = space
        self._sample_marginal = sample_marginal
        self._sample_conditional = sample_conditional

    @property
    def sample_conditional(self):
        if self._sample_conditional is None:
            raise RuntimeError("sample_conditional was not set")
        return lambda *args, **kwargs: self._sample_conditional(
            self.space, *args, **kwargs
        )

    @sample_conditional.setter
    def sample_conditional(self, value: Callable):
        assert callable(value)
        self._sample_conditional = value

    @property
    def sample_marginal(self):
        if self._sample_marginal is None:
            raise RuntimeError("sample_marginal was not set")
        return lambda *args, **kwargs: self._sample_marginal(
            self.space, *args, **kwargs
        )

    @sample_marginal.setter
    def sample_marginal(self, value: Callable):
        assert callable(value)
        self._sample_marginal = value

    @property
    def dim(self):
        return self.space.dim


class ProductLatentSpace(LatentSpace):
    """A latent space which is the cartesian product of other latent spaces."""

    def __init__(self, spaces: List[LatentSpace], content_dependent_style=False):
        self.spaces = spaces
        self.content_dependent_style = content_dependent_style

    def sample_conditional(self, z, size, **kwargs):
        x = []
        n = 0
        for s in self.spaces:
            if len(z.shape) == 1:
                z_s = z[n : n + s.space.n]
            else:
                z_s = z[:, n : n + s.space.n]
            n += s.space.n
            x.append(s.sample_conditional(z=z_s, size=size, **kwargs))

        return torch.cat(x, -1)

    def sample_marginal(self, size, **kwargs):
        x = [s.sample_marginal(size=size, **kwargs) for s in self.spaces]
        if self.content_dependent_style:
            B = torch.randn(x[1].size(-1), x[0].size(-1), device=x[0].device)
            a = torch.randn(x[1].size(-1), device=x[0].device)
            content_dependent_style = torch.einsum("ij,kj -> ki", B, x[0]) + a
            x[1] += content_dependent_style
        return torch.cat(x, -1)

    @property
    def dim(self):
        return sum([s.dim for s in self.spaces])