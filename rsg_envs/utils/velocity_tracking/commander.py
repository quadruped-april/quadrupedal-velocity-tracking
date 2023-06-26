import numpy as np

__all__ = ['make_commander']


class Commander:
    def read_command(self):
        raise NotImplementedError

    @property
    def normalized(self) -> bool:
        raise NotImplementedError


class FixedCommander(Commander):
    def __init__(self, cmd):
        self.cmd = np.array(cmd)

    def read_command(self):
        return self.cmd

    @property
    def normalized(self):
        return False


class StepCommander(Commander):
    def __init__(self, step_len):
        self.step_len = step_len
        self.idx = 0

    def read_command(self):
        self.idx += 1
        step = self.idx // self.step_len * 0.1
        return (
            max(min(step, 2.0 - step), 0.0),
            0., 0.
        )

    @property
    def normalized(self):
        return True


class GamepadCommander(Commander):
    def __init__(self, gamepad_type='Xbox'):
        from thirdparty.gamepad import gamepad, controllers
        if not gamepad.available():
            raise EOFError('Gamepad not found')
        try:
            self.gamepad: gamepad.Gamepad = getattr(controllers, gamepad_type)()
            self.gamepad_type = gamepad_type
        except AttributeError:
            raise RuntimeError(f'`{gamepad_type}` is not supported, '
                               f'all {controllers.all_controllers}')
        self.gamepad.startBackgroundUpdates()
        print('Gamepad connected')

    @classmethod
    def is_available(cls):
        from thirdparty.gamepad import gamepad
        return gamepad.available()

    def read_command(self):
        if self.gamepad.isConnected():
            x_speed = -self.gamepad.axis('LAS -Y')
            y_speed = -self.gamepad.axis('LAS -X')
            steering = -self.gamepad.axis('RAS -X')
            return x_speed, y_speed, steering
        else:
            raise EOFError('Gamepad disconnected')

    def __del__(self):
        self.gamepad.disconnect()

    @property
    def normalized(self):
        return True


def make_commander(cmder_type, env_cfg, verbose=True) -> Commander:
    if cmder_type == 'step':
        dt = env_cfg['control_dt']
        commander = StepCommander(int(2.0 / dt))
    elif cmder_type is not None:
        commander = FixedCommander(eval(cmder_type))
    elif GamepadCommander.is_available():
        commander = GamepadCommander()
    else:
        commander = None
        if verbose:
            print('Gamepad not found, use random command')
    if commander is not None:
        env_cfg['random_command'] = False
    return commander
