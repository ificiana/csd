import time

SIM_SPEED = 1
TIME_STEP = 0.001
SIM_TIME = 0
START_TIME = time.time_ns()


def get_time():
    return SIM_TIME


def time_tick():
    global SIM_TIME
    SIM_TIME += 1
    return SIM_TIME
