# read sensor data and make a web app to navigate the data
# fastapi websockets

# ability to analyse live data as it streams, as well as select a past run from data/
# streaming data would have less features to keep up with the polling rate

import time

from tel import MMapJSON

SENSOR_CHANNELS = [
    "channel/cube.mmap",
    "channel/drone_0.mmap",
    "channel/drone_1.mmap",
    "channel/drone_2.mmap",
    "channel/drone_3.mmap",
]

DAT = MMapJSON(f"data/{time.time()}.mmap")
sensors = {name: MMapJSON(name) for name in SENSOR_CHANNELS}
data = []
POLLING_RATE = 50
last_sim_time = None

while True:
    d = {}
    sim_time = None

    for name, reader in sensors.items():
        try:
            payload = reader.read()
            if payload:
                d[name] = payload
                if sim_time is None and "sim_time" in payload:
                    sim_time = payload["sim_time"]
            else:
                d[name] = None
        except Exception as e:
            print(f"[WARN] Failed to read {name}: {e}")
            d[name] = None

    # deduplicate based on sim_time
    if sim_time is not None and sim_time != last_sim_time:
        data.append(d)
        DAT.write(data)
        last_sim_time = sim_time
    else:
        pass

    time.sleep(1 / POLLING_RATE)

# [channel/cube.mmap] {'irl_time': 19.2749271, 'sim_time': 7.418000000000812, 'pos': [0.0, 0.0, 0.5018259829537246], 'rot': [[0.9999999937678251, 0.0001027667855207696, 4.362726301503014e-05], [-0.00010292383714743764, 0.9999934596947795, 0.003615241949012248], [-4.325545088543424e-05, -0.003615246416766741, 0.9999934640397969]], 'acc': [0.0, 0.0, 1.0899135531872162e-06], 'vel': [0.0, 0.0, -44.29214934566758], 'ang_acc': [41.0065658569336, 41.15626831054688, 0.0], 'ang_vel': [0.12435731631573049, 0.878421786218241, 0.0]}
# [channel/drone_0.mmap] {'irl_time': 19.2757026, 'sim_time': 7.419000000000812, 'thrust': [0.0, 0.0, 0.0]}
# [channel/drone_1.mmap] {'irl_time': 19.2760891, 'sim_time': 7.419000000000812, 'thrust': [0.0, 0.0, 0.0]}
# [channel/drone_2.mmap] {'irl_time': 19.2764911, 'sim_time': 7.419000000000812, 'thrust': [0.0, 0.0, 0.0]}
# [channel/drone_3.mmap] {'irl_time': 19.2770453, 'sim_time': 7.419000000000812, 'thrust': [0.0, 0.0, 0.0]}