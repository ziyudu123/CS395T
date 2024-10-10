import numpy as np
from numba import njit, prange, config
from time import time


config.THREADING_LAYER = 'threadsafe'

@njit(parallel = True)
def scan_cpu(x):
    n = len(x)
    l = np.log2(n)
    assert np.ceil(l) == np.floor(l)
    l = int(l)
    
    for il in range(l):
        dk = 2**il
        for i in prange(1, 1 + int(n/2**(il + 1))):
            k = i*2*dk - 1
            x[k] = x[k] + x[k - dk]
            
    for il in range(l - 2, -1, -1):
        dk = 2**il
        for i in prange(1, int(n/2**(il + 1))):
            k = i*2*dk - 1
            x[k + dk] = x[k + dk] + x[k]
            
if __name__ == '__main__':
    x = np.ones(2**28)
    x0 = x.copy()
    
    tic = time()
    scan_cpu(x)
    toc = time()
    print('time: ', toc - tic)
    
    tic = time()
    xx = np.cumsum(x0)
    toc = time()
    print('error: ', np.linalg.norm(x - xx))
    print('time of np.cumsum: ', toc - tic)