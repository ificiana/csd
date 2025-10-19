import time

from clock import START_TIME, get_time
from tel import MMapJSON


class Entity:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.transmitter = MMapJSON(f"channel/{topic}")

    def transmit(self):
        self.transmitter.write(self.telemetry)

    @property
    def telemetry(self):
        return {
            "irl_time": (time.time_ns() - START_TIME) / 1e9,
            "sim_time": get_time(),
        }
