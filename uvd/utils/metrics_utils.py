import numpy as np

__all__ = ["simlog"]


def simlog(x):
    return np.sign(x) * np.log(np.abs(x) + 1)
