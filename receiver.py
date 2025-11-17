"""
Telemetry receiver and data logging.

Polls simulation telemetry channels and aggregates time-series data
into a single memory-mapped file for visualization and analysis.
"""

import time

from config import (
    DRONE_TELEMETRY_FIELDS,
    PAYLOAD_TELEMETRY_FIELDS,
    POLLING_RATE,
    SENSOR_CHANNELS,
)
from tel import MMapJSON

DEL = True

DAT = MMapJSON(f"data/{int(time.time())}.mmap", file=True)
sensors = {name: MMapJSON(f"channel/{name}") for name in SENSOR_CHANNELS}

if DEL:
    [s.clear() for s in sensors.values()]

data = {
    **{f"drone_{k}": {field: {} for field in DRONE_TELEMETRY_FIELDS} for k in range(4)},
    "cube": {field: {} for field in PAYLOAD_TELEMETRY_FIELDS},
}

while True:
    for name, reader in sensors.items():
        try:
            payload = reader.read()
            if payload:
                t = payload["sim_time"]
                for k, v in payload.items():
                    if k == "sim_time":
                        continue
                    else:
                        data[name][k][t] = v
        except Exception as e:
            print(f"[WARN] Failed to read {name}: {e}")

    DAT.write(data)
    time.sleep(1 / POLLING_RATE)
