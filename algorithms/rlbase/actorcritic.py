import torch
from torch import nn

from .distribution import *
from ..utils import MLP, NetworkFactory
from ..utils.recurrent import *

__all__ = ['GeneralActor', 'InferenceActor']


class GeneralActor(nn.Module):
    def __init__(self, preprocess: MLP | RMlp, distribution: Distribution = None):
        super().__init__()

        self.preprocess = preprocess
        if distribution is None:
            distribution = Gaussian(self.preprocess.output_shape[-1])
        self.distribution = distribution
        self.is_recurrent = is_recurrent(self.preprocess)
        if self.is_recurrent:
            self.forward = self.forward_recurrent

    @classmethod
    def make(cls, cfg, ob_dim, action_dim):
        cfg = cfg.copy()
        factory = NetworkFactory.make(cfg, ob_dim, action_dim)
        distribution = Gaussian(action_dim)
        return cls(factory.nn, distribution)

    def explore(self, obs, h=None, sample=True):
        if self.is_recurrent:
            feature, hidden = self.preprocess(obs, h)
        else:
            feature, hidden = self.preprocess(obs), None
        action_mean, action_std = self.distribution(feature)
        if not sample:
            return (action_mean, action_std), hidden
        action, log_prob = self.distribution.sample_log_prob(action_mean, action_std)
        return (action_mean, action_std), (action, log_prob), hidden

    def act_log_prob(self, obs, *args, **kwargs):
        _, (action, log_prob), _ = self.explore(obs, *args, **kwargs, sample=True)
        return action, log_prob

    def act_stochastic(self, obs, *args, **kwargs):
        _, (action, _), hidden = self.explore(obs, *args, **kwargs, sample=True)
        return action, hidden

    def calc_log_prob_entropy(self, mean, std, sample):
        log_prob = self.distribution.calc_log_prob(mean, std, sample)
        entropy = self.distribution.calc_entropy(std)
        return log_prob, entropy

    def forward(self, obs, *args, **kwargs):
        feature = self.preprocess(obs)
        return self.distribution.determine(feature), None

    def forward_recurrent(self, obs, h=None):
        feature, hidden = self.preprocess(obs, h)
        return self.distribution.determine(feature), hidden

    @property
    def obs_shape(self):
        return self.preprocess.input_shape

    @property
    def action_shape(self):
        return self.distribution.action_shape

    def set_exploration_std(self, std):
        if isinstance(self.distribution, Gaussian):
            self.distribution.set_std(std)

    def clamp_exploration_std(self, min=None, max=None, indices=None):
        if isinstance(self.distribution, Gaussian):
            self.distribution.clamp_std(min, max, indices)

    def state_dict(self):
        return {
            'preprocess': self.preprocess.state_dict(),
            'distribution': self.distribution.state_dict(),
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        if not strict:
            raise ValueError('Strict mode is allowed only.')
        try:
            self.preprocess.load_state_dict(state_dict['preprocess'], strict)
            if type(self.distribution) is not Distribution:
                self.distribution.load_state_dict(
                    state_dict['distribution'],
                    strict
                )
        except KeyError:
            print('GeneralActor: Try loading a state_dict of old type.')
            self.preprocess.load_state_dict(state_dict['architecture'], strict)
            self.distribution.load_state_dict(state_dict['distribution'], strict)

    def restore(self, model_path, prefix='actor'):
        device = next(self.parameters()).device
        state_dict = torch.load(model_path, map_location=device)
        if prefix is not None:
            state_dict = state_dict[prefix]
        self.load_state_dict(state_dict)
        return self

    def inference(self):
        return InferenceActor(self)


class InferenceActor(nn.Module):
    def __init__(self, actor: GeneralActor):
        super().__init__()
        self.actor = actor
        self._hidden = None
        param = next(self.actor.parameters())
        self._datatype, self._device = param.dtype, param.device

    @torch.inference_mode()
    def forward(self, x, *, stochastic=False):
        x = torch.as_tensor(x, dtype=self._datatype, device=self._device)
        if stochastic:
            y, self._hidden = self.actor.act_stochastic(x, self._hidden)
        else:
            y, self._hidden = self.actor(x, self._hidden)
        return y.cpu().numpy()

    @torch.inference_mode()
    def reset(self, indices=None):
        if indices is None or self._hidden is None:
            self._hidden = None
            return
        if isinstance(self._hidden, torch.Tensor):  # gru
            self._hidden[..., indices, :] = 0.0
        else:
            for hidden in self._hidden:  # lstm
                hidden[..., indices, :] = 0.0

