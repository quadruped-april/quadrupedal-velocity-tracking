import json
import socket
import time

import numpy as np

from .sprint import sprint

__all__ = ['DataStream']


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class DataStream(object):
    def __init__(self, visualizer='PlotJuggler', dump=None):
        self._publisher = make_publisher(visualizer)
        self._task = None
        self._count = 0
        self._offset = None
        self._dump = dump
        self._history_data = []

    def __del__(self):
        if self._dump:
            sprint.bG(f"Dumping locomotion data to {self._dump} ...")
            with open(self._dump, 'w') as f:
                json.dump(self._history_data, f, cls=NumpyJsonEncoder)

    def set_timestamp_offset(self, offset):
        self._offset = offset

    def publish(self, data, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        elif self._offset is not None:
            timestamp += self._offset
        data = data | {'stamp': timestamp}
        self._publisher.send(data)
        if self._dump:
            self._history_data.append(data)


class Publisher:
    def __init__(self, *args, **kwargs):
        pass

    def send(self, data: dict):
        pass


def make_publisher(visualizer, *args, **kwargs) -> Publisher:
    visualizer = visualizer.lower()
    if visualizer == 'plotjuggler':
        return UdpPublisher(*args, **kwargs)
    raise ValueError(f'Unknown visualizer `{visualizer}`')


class UdpPublisher(Publisher):
    """
    Send data stream of locomotion to outer tools such as PlotJuggler.
    """

    def __init__(self, port=9870):
        super().__init__()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.port = port

    def send(self, data: dict):
        msg = json.dumps(data, cls=NumpyJsonEncoder)
        ip_port = ('127.0.0.1', self.port)
        self.server.sendto(msg.encode('utf-8'), ip_port)

