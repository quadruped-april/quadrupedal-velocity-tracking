import math
import torch
from torch import nn

LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))
ENTROPY_BIAS = 0.5 + 0.5 * math.log(2 * math.pi)

__all__ = ['Distribution', 'Gaussian', 'make_distribution']


class Distribution(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    @property
    def action_shape(self):
        return [self.action_dim]

    @torch.jit.export
    def determine(self, x):
        return x

    def forward(self, x):
        return x, None

    def sample(self, mean, std):
        return mean

    def sample_log_prob(self, mean, std):
        return mean, None

    def calc_log_prob(self, mean, std, sample):
        raise NotImplementedError

    def calc_entropy(self, std):
        raise NotImplementedError


class GaussianBase(Distribution):
    def __init__(self, action_dim, squash=False):
        super().__init__(action_dim)
        self.squash = squash

    def sample(self, mean, std):
        sample = torch.distributions.Normal(mean, std).rsample()
        return sample.tanh() if self.squash else sample

    def sample_log_prob(self, mean, std):
        sample = torch.distributions.Normal(mean, std).rsample()
        log_prob = self._gaussian_log_prob(mean, std, sample)
        if self.squash:
            sample = sample.tanh()
            log_prob = log_prob - torch.log(
                (1 - sample.pow(2)) + 1e-8
            ).sum(dim=-1, keepdim=True)
        return sample, log_prob

    def calc_log_prob(self, mean, std, sample):
        if self.squash:
            raise NotImplementedError
        return self._gaussian_log_prob(mean, std, sample)

    @staticmethod
    def _gaussian_log_prob(mean, std, sample):
        return torch.sum(
            -((sample - mean) ** 2) / (2 * std ** 2) - std.log() - LOG_SQRT_2PI,
            dim=-1, keepdim=True
        )

    def calc_entropy(self, std):
        if self.squash:
            raise NotImplementedError
        return torch.sum(
            ENTROPY_BIAS + std.log(),
            dim=-1, keepdim=True
        )


class Gaussian(GaussianBase):
    def __init__(self, action_dim, squash=False, init_std=1.0):
        super().__init__(action_dim, squash=squash)
        self.std = nn.Parameter(init_std * torch.ones(action_dim))

    def forward(self, x):
        return x, self.std.repeat(*x.shape[:-1], 1)

    def set_std(self, std):
        self.std.data[:] = std

    def clamp_std(self, min=None, max=None, indices=None):
        self.std.data[indices].clamp_(min=min, max=max)


def make_distribution(
    action_dim,
    squash=False,
):
    return Gaussian(action_dim, squash=squash)
