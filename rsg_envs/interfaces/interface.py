import os
import time
from collections import defaultdict

import numpy as np


class EnvWrapper:
    def __init__(self, impl, seed=None, normalize_ob=True):
        self._core = impl
        self.seed(seed)
        self._core.init()
        self._ob_dim = self._core.getObDim()
        self._normalize_ob = normalize_ob
        if normalize_ob:
            self._ob_mean = np.zeros(self._ob_dim, dtype=np.float32)
            self._ob_std = np.ones(self._ob_dim, dtype=np.float32)
        self._ob = np.zeros(self._ob_dim, dtype=np.float32)
        self._reward_summary = defaultdict(lambda: 0)

    @property
    def ob_dim(self):
        return self._ob_dim

    @property
    def action_dim(self):
        return self._core.getActionDim()

    def set_scaling(self, mean, var, eps=1e-8):
        self._ob_mean = mean
        self._ob_std = np.sqrt(var + eps)

    def load_scaling(self, dir_name):
        if not self._normalize_ob:
            raise RuntimeError("Observation normalization is off")
        data = np.load(os.path.join(dir_name, f'scaling.npz'))
        self.set_scaling(data['mean'], data['var'], data.get('eps', 1e-8))

    def step(self, action: np.ndarray):
        reward = self._core.step(action.squeeze())
        done = self._core.isTerminalState()
        timeout = self._core.isTimeOut()
        ob = self._observe(self._normalize_ob)
        reward_info = self._core.getRewardInfo()
        for k, v in reward_info.items():
            self._reward_summary[k] += v
        info = {
            'timeout': timeout,
            'raw_ob': self._ob,
            'reward_info': reward_info,
        }
        info.update(self._collect_data())
        return ob, reward, done, info

    def substeps(self, action: np.ndarray):
        self._core.prestep(action.squeeze())
        num_substeps = self._core.getNumSubsteps()
        for i in range(num_substeps):
            self._core.substep(i)
            yield self._collect_data_substep()

    def poststep(self, contains_substep_info=False):
        reward = self._core.poststep()
        done = self._core.isTerminalState()
        timeout = self._core.isTimeOut()
        ob = self._observe(self._normalize_ob)
        reward_info = self._core.getRewardInfo()
        for k, v in reward_info.items():
            self._reward_summary[k] += v
        info = {
            'timeout': timeout,
            'raw_ob': self._ob,
            'reward_info': reward_info,
        }
        info.update(self._collect_data_step())
        if contains_substep_info:
            info.update(self._collect_data_substep())
        return ob, reward, done, info

    def _collect_data_substep(self):
        return {
            'AngularVel': self._core.getAngularVelocity(),
            'BodyPos': self._core.getBodyPosition(),
            'ContactBuf': self._core.getContactBuf(),
            'GeneralizedForce': self._core.getGeneralizedForce(),
            'GeneralizedVel': self._core.getGeneralizedVelocity(),
            'JointPos': self._core.getJointPosition(),
            'JointVel': self._core.getJointVelocity(),
            'LinearAcc': self._core.getLinearAccel(),
            'LinearVel': self._core.getLinearVelocity(),
            'TarJointPos': self._core.getTargetJointPos(),
            'TarJointVel': self._core.getTargetJointVel(),
            'Torque': self._core.getJointTorque(),
            'TorqueDes': self._core.getDesiredTorque(),
        }

    def _collect_data_step(self):
        return {
            'ActionFluc': abs(self._core.getActionFluc()),
            'AirTime': self._core.getAirTime(),
            'Clearance': self._core.getClearance(),
            'ContactEdge': self._core.getContactEdge(),
            'ContactStates': self._core.getContactStates(),
            'FootForces': self._core.getFootForces(),
            'GaitPeriod': self._core.getGaitPeriod(),
            'GeneralizedAccel': self._core.getGeneralizedAccel(),
            'Power': self._core.getPowerSum(),
            'Slip': self._core.getSlip(),
            'Rpy': self._core.getRpy(),
        }

    def _collect_data(self):
        return self._collect_data_substep() | self._collect_data_step()

    def seed(self, seed=None):
        if seed is None:
            seed = time.time_ns()
        self._core.seed(seed)

    def reset(self):
        self._core.reset()
        return self._observe(self._normalize_ob)

    def _observe(self, normalize=True):
        self._core.observe(self._ob)
        ob = self._ob.copy()
        if normalize:
            ob = (ob - self._ob_mean) / self._ob_std
        return ob

    def get_reward_summary(self):
        return self._reward_summary

    def __getattr__(self, item):
        return getattr(self._core, item)
