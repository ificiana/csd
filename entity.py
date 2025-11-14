"""
Base entity class for simulation objects with telemetry.

Provides standardized telemetry transmission via memory-mapped JSON channels.
Each entity maintains its own communication topic for real-time data sharing.
"""

import time

from clock import START_TIME, get_time
from tel import MMapJSON


class Entity:
    """Base class for simulation entities with telemetry capabilities."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.transmitter = MMapJSON(f"channel/{topic}")

    def transmit(self):
        """Sends current telemetry data to the entity's channel."""
        self.transmitter.write(self.telemetry)

    @property
    def telemetry(self) -> dict[str, float | list[float]]:
        """Returns basic timing telemetry. Override in subclasses."""
        return {
            "irl_time": (time.time_ns() - START_TIME) / 1e9,
            "sim_time": get_time(),
        }
