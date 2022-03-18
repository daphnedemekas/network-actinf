import numpy as np

EPS = 1e-16

def log_stable(arr):

    return np.log(arr + EPS)