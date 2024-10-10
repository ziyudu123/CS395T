import numpy as np
import cupy as cp
from numba import cuda
from mpi4py import MPI
from time import time
import argparse

from scan_cpu_gpu_mpi import scan_cpu_gpu

prsr = argparse.ArgumentParser(description = 'Test scan')
prsr.add_argument('-l', dest = 'l', type = int)
args = prsr.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_gpus = len(cuda.gpus)
cp.cuda.Device(rank%num_gpus).use()

# run once before test
print('start warming up')
x_temp = np.ones(1024)
scan_cpu_gpu(x_temp, comm)
print('warmed up')

N = 2**args.l
n = int(N/size)

print('l: ', args.l)
print('size: ', size)

if rank == 0:
    x = np.ones(N)
else:
    x = None

x_local = np.zeros(n)
comm.Scatter(x, x_local, root = 0)

comm.barrier()

tic = time()
scan_cpu_gpu(x_local, comm)
toc = time()
print('rank: ', rank)
print('time: ', toc - tic)

comm.Gather(x_local, x, root = 0)


if rank == 0:
    print('result: ', x)
    ind = np.random.randint(0, N, 10000)
    print('ind_test: ', ind + 1)
    print('x_ind: ', x[ind])
    print('error: ', np.linalg.norm(ind + 1 - x[ind]))
