# read sensor data and make a web app to navigate the data
# fastapi websockets

import time

from tel import MMapJSON

SENSOR_CHANNELS = [
    "channel/cube.mmap",
    "channel/drone_0.mmap",
    "channel/drone_1.mmap",
    "channel/drone_2.mmap",
    "channel/drone_3.mmap",
]

sensors = {name: MMapJSON(name) for name in SENSOR_CHANNELS}

while True:
    for name, reader in sensors.items():
        try:
            payload = reader.read()
            if payload:
                print(f"[{name}] {payload}")
        except Exception as e:
            print(f"[WARN] Failed to read {name}: {e}")
    exit()
    time.sleep(1)
