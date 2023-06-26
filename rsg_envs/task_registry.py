import os.path
from dataclasses import dataclass
from typing import Any, Iterable

import yaml

from .interfaces import EnvWrapper

__all__ = ['Task', 'registry', 'HOME_PATH']

HOME_PATH = os.path.realpath(f'{__file__}/../..')
_RSC_PATH = os.path.join(HOME_PATH, 'resources')


@dataclass
class Task(object):
    name: str
    Env: Any
    cfgs: tuple[str] = ()
    cfg_dir: Any = None

    def make_env(self, cfg, render, verbose, *args, wrapper=EnvWrapper, **kwargs):
        env = self.Env(_RSC_PATH, yaml.safe_dump(cfg), render, verbose)
        if wrapper is not None:
            env = wrapper(env, *args, **kwargs)
        return env

    def load_cfg(self):
        cfg = {}
        for file in reversed(self.cfgs):
            file = file.format(TASK=self.cfg_dir)
            with open(file, 'r') as f:
                cfg.update(yaml.safe_load(f))
        return cfg

    @property
    def robot(self):
        return self.name.split('-', 1)[0]


def src_join(rel):
    return os.path.join(os.path.dirname(__file__), rel)


def cfg_join(rel):
    return os.path.join(HOME_PATH, 'configs', rel)


class _TaskRegistry(object):
    __instance = None
    __module = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        self._tasks: dict[str, Task] = {}

    def _register(
        self,
        name: str,
        cls: str,
        cfgs: Iterable[str] = (),
        cfg_dir: str = None,
        robot=None,
    ):
        if robot is None:
            robot, _ = name.split('-', 1)
        try:
            robot_ns = getattr(self.__module, robot)
            env_cls = getattr(robot_ns, cls)
        except AttributeError:
            return False

        self._tasks[name] = Task(
            name, env_cls, tuple(cfgs), cfg_dir,
        )
        return True

    def list_envs(self):
        if self.__module is None:
            self.__load_envs()
        return list(self._tasks.keys())

    def get(self, robot, name=None) -> Task:
        if self.__module is None:
            self.__load_envs()
        name = robot if name is None else f'{robot}-{name}'
        if name not in self._tasks:
            raise ValueError(
                f'No environment named {name}, '
                f'all: {list(self._tasks.keys())}'
            )
        return self._tasks[name]

    def __load_envs(self):
        from rsg_envs.bin import rsg_envs as envs
        self.__module = envs

        self._register(
            'aliengo-velocity-tracking', 'VelocityTracking',
            ['{TASK}/env.yml'], cfg_dir=cfg_join('aliengo'),
        )


registry = _TaskRegistry()
