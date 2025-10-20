import time

TIME_STEP = 0.001
SIM_TIME = 0
START_TIME = time.time_ns()

# when to end the sim
T = 10


def get_time():
    return SIM_TIME


def time_tick():
    global SIM_TIME
    SIM_TIME += TIME_STEP
    return SIM_TIME
