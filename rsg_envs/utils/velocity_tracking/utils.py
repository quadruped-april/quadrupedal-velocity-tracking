import os
import re
from datetime import datetime

import yaml

__all__ = ['supports_symmetry', 'RunParser']


def supports_symmetry(task):
    return (
        hasattr(task.Env, 'ObsSymmetry') and
        hasattr(task.Env, 'ActSymmetry')
    )


class RunParser(object):
    def __init__(
        self,
        weight_path,
        pattern='full_(?P<iter>[0-9]+).pt',
        datetime_fmt="%Y-%m-%d-%H-%M-%S",
    ):
        self._path = weight_path
        if not os.path.exists(self._path):
            raise FileNotFoundError(f'{self._path} not found')
        if os.path.isdir(self._path):
            self._dir = self._path
            self._iter = -1
            for x in os.listdir(self._path):
                res = re.match(pattern, x)
                if res is not None:
                    if (it := int(res.group('iter'))) > self._iter:
                        self._iter = it
            if self._iter == -1:
                raise RuntimeError(f'Checkpoint in {self._path} not found.')
            filename = pattern.replace('(?P<iter>[0-9]+)', str(self._iter))
            self._path = os.path.join(self._dir, filename)
        else:
            filename = os.path.basename(weight_path)
            res = re.match(pattern, filename)
            if res is None:
                raise RuntimeError(f'Filename `{filename}` does not match pattern {pattern}.')
            self._iter = int(res.group('iter'))
            self._dir = os.path.dirname(weight_path)
        self._path = os.path.abspath(self._path)
        self._dir = os.path.abspath(self._dir)
        self._task_name = os.path.basename(os.path.dirname(self._dir))
        try:
            from rsg_envs import registry
            self._task = registry.get(self._task_name)
        except (ValueError, ModuleNotFoundError):
            self._task = None
        self._run_info_str = os.path.basename(self._dir)
        try:
            self._datetime_str, self._run_name = self._run_info_str.split(':', 1)
            self._datetime = datetime.strptime(self._datetime_str, datetime_fmt)
        except ValueError:
            self._run_name = self._run_info_str
            self._datetime_str = 'Unknown'
            self._datetime = None

    model_path = property(lambda self: self._path)
    directory = property(lambda self: self._dir)
    iteration = property(lambda self: self._iter)
    task = property(lambda self: self._task)
    task_name = property(lambda self: self._task_name)
    datetime_str = property(lambda self: self._datetime_str)
    datetime = property(lambda self: self._datetime)
    run_name = property(lambda self: self._run_name)

    @property
    def brief(self):
        return f'{self._task_name}/{self._run_name}/{self._iter}'

    def __str__(self):
        return (
            f'run\n'
            f'- task: {self._task_name}\n'
            f'- name: {self._run_name}\n'
            f'- start at: {self._datetime or self._datetime_str}\n'
            f'- iteration: {self._iter}'
        )

    def join(self, filename):
        return os.path.join(self._dir, filename)

    @property
    def scaling_path(self):
        scaling_path = self.join(f'scaling_{self._iter}.npz')
        return scaling_path if os.path.exists(scaling_path) else None
