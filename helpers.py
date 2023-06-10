import os
import datetime
import _pickle as cp
import json
import logging
import numpy as np
import threading
import pandas as pd
from pytz import timezone, utc


PROJECT_DIR = os.path.abspath(os.path.join(__file__, '../../../..'))

THREAD_DICT = {}

FORMAT_STR = "%(asctime)s.%(msecs)03d [%(module)s] %(filename)s:%(lineno)s %(levelname)s: %(message)s"


def custom_tz(*args):
    utc_dt = utc.localize(datetime.datetime.utcnow())
    my_tz = timezone('Asia/Hong_Kong')
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()


class MyFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.format_str = FORMAT_STR
        self.date_fmt = '%Y-%m-%d %H:%M:%S'

    def format(self, record):
        log_fmt = self.format_str
        formatter = logging.Formatter(log_fmt, self.date_fmt)
        return formatter.format(record)


def get_logger(name):
    logger = logging.getLogger(name)
    logging.Formatter.converter = custom_tz
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(MyFormatter())
    logger.addHandler(ch)
    logger.propagate = False
    return logger


helper_logger = get_logger('helper')
log_info = helper_logger.info
log_error = helper_logger.error


def read_json_func(read_path, **kwargs):
    with open(read_path, 'r') as f:
        res_dict = json.load(f, **kwargs)
    return res_dict


def save_json_func(save_json, save_path, **kwargs):
    with open(save_path, 'w') as f:
        json.dump(save_json, f, cls=NpEncoder, **kwargs)
    return


def read_pkl_helper(read_path, **kwargs):
    with open(read_path, 'rb') as f:
        res = cp.load(f, **kwargs)
    return res


def save_pkl_helper(save_pkl, save_path, **kwargs):
    with open(save_path, 'wb') as f:
        cp.dump(save_pkl, f, **kwargs)
    return


def thread_run_func(func, kwargs):
    # Write a thread running utility that doesn't block the main function
    return


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def error_str(e):
    return type(e).__name__ + ': ' + str(e)
