import time

from config import TIME_STEP

SIM_TIME = 0
START_TIME = time.time_ns()


def get_time():
    return SIM_TIME


def time_tick():
    global SIM_TIME
    SIM_TIME += TIME_STEP
    return SIM_TIME
