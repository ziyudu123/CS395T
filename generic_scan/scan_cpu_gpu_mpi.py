import numpy as np
import cupy as cp
from time import time
from mpi4py import MPI

from scan_cpu import scan_cpu
from scan_gpu import scan_gpu


def scan_cpu_gpu(x, comm):
    # x local
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n = len(x)
    n_half = int(n/2)
    
    x_cpu = x[:n_half]
    x_gpu = cp.asarray(x[n_half:])
    
    tic = time()
    scan_cpu(x_cpu)
    toc = time()
    print('time_scan_cpu: ', toc - tic)
    
    tic = time()
    scan_gpu(x_gpu)
    toc = time()
    print('time_scan_gpu: ', toc - tic)
    
    x_cpu_sum = x_cpu[n_half - 1]
    x_sum = x_cpu_sum + cp.asnumpy(x_gpu[n_half - 1])
    x_gpu += x_cpu_sum
    
    x_sum_global = np.zeros(1, dtype = np.float64)
    comm.Scan(x_sum, x_sum_global, op = MPI.SUM)
    
    rank_send = (rank + 1)%size
    rank_recv = (rank - 1)%size
    buff_send = x_sum_global[0]
    req_send = comm.isend(buff_send, dest = rank_send)
    req_recv = comm.irecv(source = rank_recv)
    buff_recv = req_recv.wait()
    
    if rank > 0:
        x_cpu += buff_recv
        x_gpu += buff_recv
        
    x[n_half:] = cp.asnumpy(x_gpu)
    
    req_send.wait()
    

    
        
    
    
    