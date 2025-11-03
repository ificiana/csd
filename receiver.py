import time

from tel import MMapJSON

DEL = True

SENSOR_CHANNELS = [
    "cube",
    "drone_0",
    "drone_1",
    "drone_2",
    "drone_3",
]

DAT = MMapJSON(f"data/{int(time.time())}.mmap", file=True)
sensors = {name: MMapJSON(f"channel/{name}") for name in SENSOR_CHANNELS}

if DEL:
    [s.clear() for s in sensors.values()]

data = {
    **{
        f"drone_{k}": {
            "irl_time": {},
            "thrust": {},
            "command": {},
            "pos": {},
        }
        for k in range(4)
    },
    "cube": {
        "irl_time": {},
        "pos": {},
        "rot": {},
        "acc": {},
        "vel": {},
        "ang_acc": {},
        "ang_vel": {},
    },
}

POLLING_RATE = 50

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

# [channel/cube.mmap] {'irl_time': 19.2749271, 'sim_time': 7.418000000000812, 'pos': [0.0, 0.0, 0.5018259829537246], 'rot': [[0.9999999937678251, 0.0001027667855207696, 4.362726301503014e-05], [-0.00010292383714743764, 0.9999934596947795, 0.003615241949012248], [-4.325545088543424e-05, -0.003615246416766741, 0.9999934640397969]], 'acc': [0.0, 0.0, 1.0899135531872162e-06], 'vel': [0.0, 0.0, -44.29214934566758], 'ang_acc': [41.0065658569336, 41.15626831054688, 0.0], 'ang_vel': [0.12435731631573049, 0.878421786218241, 0.0]}
# [channel/drone_0.mmap] {'irl_time': 19.2757026, 'sim_time': 7.419000000000812, 'thrust': [0.0, 0.0, 0.0]}
# [channel/drone_1.mmap] {'irl_time': 19.2760891, 'sim_time': 7.419000000000812, 'thrust': [0.0, 0.0, 0.0]}
# [channel/drone_2.mmap] {'irl_time': 19.2764911, 'sim_time': 7.419000000000812, 'thrust': [0.0, 0.0, 0.0]}
# [channel/drone_3.mmap] {'irl_time': 19.2770453, 'sim_time': 7.419000000000812, 'thrust': [0.0, 0.0, 0.0]}
