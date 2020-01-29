import numpy as np

def eye(n1, n2):
    left = np.diagflat(np.ones(n1, order='F'))
    right = np.zeros((n1, n2-n1), order='F')
    return np.concatenate((left, right), axis=1)