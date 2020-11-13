import requests
import time
import torch
import random

import numpy as np
from tqdm.autonotebook import tqdm

from multiprocessing import Process
from scipy.spatial import distance

from typing import *

from ml.tspsolver import DefaultTSPSolver

if __name__ == "__main__":
    solver = DefaultTSPSolver(address="10.90.185.46", port="8000")
    solver.run()
    print("done")
