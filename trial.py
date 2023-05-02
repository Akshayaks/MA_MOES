import numpy as onp
import sys
import os

# from jax import vmap, jit, grad
# import jax.numpy as np
# from jax.lax import scan
from functools import partial
import pdb
# from sklearn.preprocessing import normalize
import common
import scalarize
from ergodic_coverage import ErgCover

import scipy.stats
import ergodic_metric
from utils import *
from explicit_allocation import *
import math
import time
import json

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from kneed import KneeLocator

print("Hello")
print("Stopped once")
print("Stopped again")