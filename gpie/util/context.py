# -*- coding: utf-8 -*-
# context managers

import colorama                                                   # type: ignore
import numpy as np                                                # type: ignore
import os
import signal
from datetime import datetime


class RandomSeed:
    """ change random seed within context """
    def __init__(self, seed: int):
        self.seed = seed
        self.state = np.random.get_state()

    def __enter__(self):
        np.random.seed(self.seed)

    def __exit__(self, *args):
        np.random.set_state(self.state)


class ChDir:
    """ change directory within context """
    def __init__(self, path: str):
        self.old_dir = os.getcwd()
        self.new_dir = path

    def __enter__(self):
        os.chdir(self.new_dir)

    def __exit__(self, *args):
        os.chdir(self.old_dir)


class Timer:
    """ record time spent within context """
    def __init__(self, prompt: str):
        self.prompt = prompt + ' takes time: '

    def __enter__(self):
        self.start = datetime.now()

    def __exit__(self, *args):
        print(self.prompt + str(datetime.now() - self.start))
        # print(colorama.Fore.BLUE + \
        #       self.prompt + str(datetime.now() - self.start)\
        #       + colorama.Style.RESET_ALL)


class TimeOut:
    """ timeout within context """
    def __init__(self, seconds: int, message='time out'):
        self.seconds = seconds
        self.message = message

    def __enter__(self):
        self.old_handle = signal.signal(signal.SIGALRM, self.handle)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *args):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_handle)

    def handle(self, *args):
        raise TimeoutError(self.message)
