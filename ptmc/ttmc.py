import tensorly as tl
import numpy as np
import torch

import os
import argparse
import random
import time

from yaml import load, Loader

from Normalizer import *

base = "../tensors/"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   UTILS 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

timers = {"tensor_norm_time":[],
          "matrix_norm_time":[],
          "tensor_recover_time":[],
          "ttm_time":[],
          "ttmc_baseline_time":[],
          "ttmc_mixed_time":[]}


precisions = { "fp16":torch.float16,
               "fp32":torch.float32,
               "fp64":torch.float64}


def read_tensor(name, u):
    tensor = []
    with open(f"{base}{name}.dns", 'r') as file:
        nu = 0
        for entry in file:
            nu += 1
            if nu==2:
                shape = list(map(int, entry.split(" ")))
            if entry.find(".")==-1:
                continue
            tensor.append(float(entry))
    return torch.tensor(device='cuda:0', data=tensor, dtype=precisions[u]).reshape(shape)


def write_tensor(filename, X):
    inds = np.array(np.nonzero(X + 1))
    with open(filename, 'w') as file:
        file.write(f"{len(X.shape)}\n{' '.join(str(idx) for idx in X.shape)}\n{X.size}\n")
        for i in range(inds.shape[1]):
            indices = inds[:, i] + 1
            line = ' '.join(str(idx) for idx in indices)
            file.write(f"{line} {X[tuple(inds[:, i])]:.18f}\n")


def write_dns(filename, X):
    with open(filename, 'w') as file:
        vals = X.flatten(order='C')
        file.write(f"{len(X.shape)}\n{' '.join([str(s) for s in X.shape])}\n{vals.size}\n")
        for i in range(len(vals)):
            if i % 10000==0:
                print(f"Writing {i}/{len(vals)}")
            file.write(f"{vals[i]:.18f}\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#             Matrix Initialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def init_matrices(args, rows, cols):
    if args.mat_init=="unif":
        return init_matrices_unif(rows, cols, args.alpha, args.out)
    else:
        raise Exception(f"Invalid init {args.mat_init}")


def init_matrices_unif(rows, cols, alpha, out):
    matrices = []
    n = len(rows)
    for i in range(n):
        matrices.append(torch.rand(rows[i], cols[i], device='cuda:0', dtype=precisions[out]))
        matrices[i] = torch.mul(matrices[i], alpha)
    return matrices

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                  Ordering
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ordering_compute(rows, cols):
    N = len(rows)
    pi = np.zeros(N)
    for i in range(N):
        pi[i] = (1 / cols[i]) - (1/rows[i])
    ordering = list(range(N))
    ordering_best = [x for _, x in sorted(zip(pi, ordering), key=lambda p: p[0])]
    print(ordering_best)
    print(pi)
    return ordering_best




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   TTMC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ttmc_mixed(X, U_list, normalizer, ordering, trial):

    tensor_norm_time = 0
    matrix_norm_time = 0
    ttm_time = 0
    tensor_recover_time = 0

    N = len(U_list)

    Y = X

    for n in range(N):

        mode = ordering[n]

        # Normalize operands
        t1 = time.time()
        Y = normalizer.normalize_tensor(Y, mode)
        tensor_norm_time += (time.time() - t1)

        t2 = time.time()
        U = normalizer.normalize_matrix(U_list[mode], mode)
        matrix_norm_time += (time.time() - t2)

        # Perform TTM
        t3 = time.time()
        Y = tl.tenalg.mode_dot(Y, U, mode)
        ttm_time += (time.time() - t3)

        # Recover output
        t4 = time.time()
        Y = normalizer.recover_tensor(Y, mode)
        tensor_recover_time += (time.time() - t4)

    if trial > 0:
        timers["tensor_norm_time"].append(tensor_norm_time)
        timers["matrix_norm_time"].append(matrix_norm_time)
        timers["ttm_time"].append(ttm_time)
        timers["tensor_recover_time"].append(tensor_recover_time)

    return Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#              Error Measurements 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def frob_norm_err(Y_correct, Y_computed):
    return (torch.norm(Y_correct - Y_computed)) / torch.norm(Y_correct)


def componentwise_err(Y_correct, Y_computed):
    err = torch.abs((Y_correct - Y_computed)) 
    return torch.max(err)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   Main 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_torch_flags(args):
    if args.accum=="fp16":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    if args.accum=="fp32":
        torch.backends.cuda.matmul.allow_tf32 = True


def arg_to_str(args):
    return


def print_times():
    for timer in timers:
        print(f"[{timer}]: {sum(timers[timer])}s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", type=str)
    parser.add_argument("--ordering", type=str)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--mat_rows", type=int, nargs='+')
    parser.add_argument("--mat_init", type=str)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--norm", type=str)
    parser.add_argument("--accum", type=str)
    parser.add_argument("--compute", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--correctness", action='store_true')

    args = parser.parse_args()

    tl.set_backend('pytorch')
    set_torch_flags(args)

    print("~"*100)
    print(f"Running {args.tensor}")

    # Set up inputs 
    X = read_tensor(args.tensor, args.out)
    U_list = init_matrices(args, args.mat_rows, X.shape)

    # Normalizer
    theta = 0.1
    normalizer = make_normalizer(args.norm, X.ndim, precisions[args.accum], precisions[args.compute], precisions[args.out], theta)


    # Run TTMc
    for t in range(args.trials):

        print(f"\tTRIAL {t}")

        ttmc_mixed_time = 0
        ttmc_baseline_time = 0

        # Determine ordering
        #ordering = ordering_compute(args.mat_rows, X.shape)
        ordering = list(range(X.ndim))
        #random.shuffle(ordering)

        t9 = time.time()
        Y_correct = tl.tenalg.multi_mode_dot(X, U_list)
        ttmc_baseline_time = (time.time() - t9)

        t0 = time.time()
        Y = ttmc_mixed(X, U_list, normalizer, ordering, t)
        ttmc_mixed_time = (time.time() - t0)

        if t > 0:
            timers["ttmc_mixed_time"].append(ttmc_mixed_time)
            timers["ttmc_baseline_time"].append(ttmc_baseline_time)

        if args.correctness:
            f_err = frob_norm_err(Y_correct, Y)
            c_err = componentwise_err(Y_correct, Y)
            print(f"||Y_hat - Y||F / ||Y_hat||F: {f_err}")
            print(f"max|Y_hat - Y|: {c_err}")

    # Forward Error

    print_times()
    print("~"*100)


