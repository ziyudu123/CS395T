import cupy as cp
from numba import cuda
import numpy as np
import math
from time import time

max_num_threads = 1024
shape_block = 2*max_num_threads
min_num_blocks = 216

@cuda.jit()
def scan_gpu_block_kernel(x, x_block_sum):
    N = len(x)
    Nb = len(x_block_sum)
    n = int(N/Nb)
    l = math.log2(n)
    assert math.ceil(l) == math.floor(l)
    l = int(l)
    
    x_block = cuda.shared.array(shape_block, dtype = np.float64)
    
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    
    idx = bx*n + 2*tx
    x_block[2*tx] = x[idx]
    x_block[2*tx + 1] = x[idx + 1]
    
    cuda.syncthreads()
    
    for il in range(l):
        dk = 2**il
        imax = 1 + int(n/2**(il + 1))
        i = tx + 1
        if i < imax:
            k = i*2*dk - 1
            x_block[k] = x_block[k] + x_block[k - dk]
            
        cuda.syncthreads()

    for il in range(l - 2, -1, -1):
        dk = 2**il
        imax = int(n/2**(il + 1))
        i = tx + 1
        if i < imax:
            k = i*2*dk - 1
            x_block[k + dk] = x_block[k + dk] + x_block[k]
        
        cuda.syncthreads()
        
    x[idx] = x_block[2*tx]
    x[idx + 1] = x_block[2*tx + 1]
    
    cuda.syncthreads()
    
    if tx == 0:
        x_block_sum[bx] = x_block[n - 1]
        
        
@cuda.jit
def scan_gpu_block_correct_kernel(x, x_correct):
    N = len(x)
    Nb = len(x_correct)
    n = int(N/Nb)
    
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    idx = bx*n + 2*tx
    
    if bx > 0:
        x_correct_block = x_correct[bx - 1]
        x[idx] = x[idx] + x_correct_block
        x[idx + 1] = x[idx + 1] + x_correct_block
        

def scan_gpu(x):
    N = len(x)
    num_threads = max_num_threads
    if N <= 2*num_threads*min_num_blocks:
        x[:] = cp.cumsum(x)
        return

    num_blocks = int(N/(2*num_threads))
    x_block_sum = cp.zeros(num_blocks, dtype = np.float64)
    scan_gpu_block_kernel[num_blocks, num_threads](x, x_block_sum)
    
    if num_blocks <= 2*num_threads*min_num_blocks:
        x_block_sum[:] = cp.cumsum(x_block_sum)
    else:
        num_blocks1 = int(num_blocks/(2*num_threads))
        x_block_sum1 = cp.zeros(num_blocks1, dtype = np.float64)
        scan_gpu_block_kernel[num_blocks1, num_threads](x_block_sum, x_block_sum1)
        x_block_sum1[:] = cp.cumsum(x_block_sum1)
        scan_gpu_block_correct_kernel[num_blocks1, num_threads](x_block_sum, x_block_sum1)
    
    scan_gpu_block_correct_kernel[num_blocks, num_threads](x, x_block_sum)
    
   
if __name__ == '__main__':
    x = cp.ones(2**28)
    x0 = x.copy()
    
    tic = time()
    scan_gpu(x)
    toc = time()
    print('results:\n', x)
    print('time: ', toc - tic)
    
    tic = time()
    xx = cp.cumsum(x0)
    toc = time()
    print('error: ', cp.linalg.norm(x - xx))
    print('time of cp.cumsum: ', toc - tic)


    
    