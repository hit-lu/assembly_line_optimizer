import copy
import time
import math
import argparse

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import wraps
from collections import defaultdict
from typing import List, Counter
from tqdm import tqdm

# 机器参数
max_head_index, max_slot_index = 6, 120     # 暂时默认所有机器参数相同
max_machine_index = 3
interval_ratio = 2
slot_interval = 15
head_interval = slot_interval * interval_ratio
head_nozzle = ['' for _ in range(max_head_index)]  # 头上已经分配吸嘴

# 位置信息
slotf1_pos, slotr1_pos = [-31.267, 44.], [807., 810.545]  # F1(前基座最左侧)、R1(后基座最右侧)位置
fix_camera_pos = [269.531, 694.823]  # 固定相机位置
anc_marker_pos = [336.457, 626.230]  # ANC基准点位置
stopper_pos = [635.150, 124.738]  # 止档块位置

# 时间参数
T_pp, T_tr, T_nc = 2, 5, 25

# 电机参数
head_rotary_velocity = 8e-5  # 贴装头R轴旋转时间
x_max_velocity, y_max_velocity = 1.4, 1.2
x_max_acceleration, y_max_acceleration = x_max_velocity / 0.079, y_max_velocity / 0.079

# 不同种类供料器宽度
feeder_width = {'SM8': (7.25, 7.25), 'SM12': (7.00, 20.00), 'SM16': (7.00, 22.00),
                'SM24': (7.00, 29.00), 'SM32': (7.00, 44.00)}

# 可用吸嘴数量限制
nozzle_limit = {'CN065': 6, 'CN040': 6, 'CN220': 6, 'CN400': 6, 'CN140': 6}


def axis_moving_time(distance, axis=0):
    distance = abs(distance) * 1e-3
    Lamax = x_max_velocity ** 2 / x_max_acceleration if axis == 0 else y_max_velocity ** 2 / y_max_acceleration
    Tmax = x_max_velocity / x_max_acceleration if axis == 0 else y_max_velocity / y_max_acceleration
    if axis == 0:
        return 2 * math.sqrt(distance / x_max_acceleration) if distance < Lamax else 2 * Tmax + (
                    distance - Lamax) / x_max_velocity
    else:
        return 2 * math.sqrt(distance / y_max_acceleration) if distance < Lamax else 2 * Tmax + (
                    distance - Lamax) / y_max_velocity


def head_rotary_time(angle):
    while -180 > angle > 180:
        if angle > 180:
            angle -= 360
        else:
            angle += 360
    return abs(angle) * head_rotary_velocity


def timer_wrapper(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)

        print("function {} running time :  {} s".format(func.__name__, time.time() - start_time))
        return result

    return measure_time


