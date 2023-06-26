import argparse

__all__ = ['make_play_argparser']


def make_play_argparser(parser=None, minimal=False) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # run loading and environment initialization
    arg('run', type=str, metavar='RUN', help='run directory or model weight path')
    arg('-r', '--robot', type=str, help='robot name (default deduced from the loaded run)')
    arg('-q', '--quiet', action='store_true', help='print less information')
    arg('--seed', type=int, help='environment seed')
    arg('--headless', action='store_true', default=False, help='disable raisim server')
    if minimal:
        return parser

    # policy config
    arg('--device', type=str, help='device for inference')

    # play config
    arg('-s', '--seconds', type=int, default=60, help='play time in seconds')
    arg('-c', '--cmd', metavar="Vx,Vy,Wz", type=str, help='fixed command in list format')
    arg('-S', '--speed', type=float, default=1., help='simulation time ratio')
    arg('-E', '--endless', action='store_true', help='play permanently')
    arg('--dump', type=str, help='dump data to a file')
    arg('--timeout', action='store_true', help='enable timeout reset')

    return parser
