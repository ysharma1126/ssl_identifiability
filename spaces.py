from abc import ABC, abstractmethod, abstractproperty
import torch
from scipy import stats
import numpy as np
from scipy.stats import wishart, truncnorm
import sys


class Space(ABC):
    @abstractmethod
    def uniform(self, size, device):
        pass

    @abstractmethod
    def normal(self, mean, std, size, device):
        pass

    @abstractmethod
    def laplace(self, mean, std, size, device):
        pass

    @abstractmethod
    def generalized_normal(self, mean, lbd, p, size, device):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass


class NRealSpace(Space):
    def __init__(self, n):
        self.n = n

    @property
    def dim(self):
        return self.n

    def uniform(self, size, device="cpu"):
        raise NotImplementedError("Not defined on R^n")

    def normal(self, mean, std, size, device="cpu", change_prob=1., statistical_dependence=False):
        """Sample from a Normal distribution in R^N.

        Args:
            mean: Value(s) to sample around.
            std: Concentration parameter of the distribution (=standard deviation).
            size: Number of samples to draw.
            device: torch device identifier
        """
        if mean is None:
            mean = torch.zeros(self.n)
        if len(mean.shape) == 1 and mean.shape[0] == self.n:
            mean = mean.unsqueeze(0)
        if not torch.is_tensor(std):
            std = torch.ones(self.n) * std
        if len(std.shape) == 1 and std.shape[0] == self.n:
            std = std.unsqueeze(0)
        assert len(mean.shape) == 2
        assert len(std.shape) == 2

        if torch.is_tensor(mean):
            mean = mean.to(device)
        if torch.is_tensor(std):
            std = std.to(device)
        change_indices = torch.distributions.binomial.Binomial(probs=change_prob).sample((size, self.n)).to(device)
        if statistical_dependence:
            Sigma = wishart.rvs(self.n, np.eye(self.n), size=1)
            assert np.all(np.linalg.eigvals(Sigma) > 0)  # check positive-definite
            changes = np.random.multivariate_normal(np.zeros(self.n), Sigma, size)
            changes = torch.FloatTensor(changes).to(device)
        else:
            changes = torch.randn((size, self.n), device=device) * std
        return mean + change_indices * changes

    def laplace(self, mean, lbd, size, device="cpu"):
        raise NotImplementedError("Not used")

    def generalized_normal(self, mean, lbd, p, size, device=None):
        raise NotImplementedError("Not used")
        
class NBoxSpace(Space):
    def __init__(self, n, min_=-1, max_=1):
        self.n = n
        self.min_ = min_
        self.max_ = max_

    @property
    def dim(self):
        return self.n

    def uniform(self, size, device="cpu"):
        raise NotImplementedError("Not used")

    def normal(self, mean, std, size, device="cpu", change_prob=1., statistical_dependence=False):
        """Sample from a Normal distribution in R^N and then restrict the samples to a box.

        Args:
            mean: Value(s) to sample around.
            std: Concentration parameter of the distribution (=standard deviation).
            size: Number of samples to draw.
            device: torch device identifier
        """

        assert len(mean.shape) == 1 or (len(mean.shape) == 2 and len(mean) == size)
        assert mean.shape[-1] == self.n

        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)

        mean = mean.to(device)
        mean_np = mean.detach().cpu().numpy()
        a = (self.min_ - mean_np) / std
        b = (self.max_ - mean_np) / std
        unnormalised_samples = truncnorm.rvs(a, b, size=(size, self.n))
        samples = torch.FloatTensor(std * unnormalised_samples + mean_np, device=device)
        return samples

    def laplace(self, mean, lbd, size, device="cpu"):
        raise NotImplementedError("Not used")

    def generalized_normal(self, mean, lbd, p, size, device=None):
        raise NotImplementedError("Not used")