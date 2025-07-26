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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#               Normalization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   TTMC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ttmc_mixed(X, U_list, normalizer, ordering):

    N = len(U_list)

    Y = X
    for n in range(N):
        print(f"TTMc iteration {n}")

        mode = ordering[n]

        Y = normalizer.normalize_tensor(Y, mode)
        U = normalizer.normalize_matrix(U_list[mode], mode)

        Y = tl.tenalg.mode_dot(Y, U, mode)

        print(torch.max(Y))
        print(Y.dtype)
        Y = normalizer.recover_tensor(Y, mode)

    return Y


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", type=str)
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
        print(f"TRIAL {t}")

        # Determine ordering
        ordering = list(range(X.ndim))
        random.shuffle(ordering)

        stime = time.time()
        Y = ttmc_mixed(X, U_list, normalizer, ordering)
        etime = time.time()
        print(f"Time for mixed: {etime-stime}")

        if args.correctness:
            stime = time.time()
            Y_correct = tl.tenalg.multi_mode_dot(X, U_list)
            etime = time.time()
            print(f"Time for normal: {etime-stime}")
            err = torch.norm(Y - Y_correct) / torch.norm(Y_correct)
            print(f"Err: {err}")

    # Forward Error


    print("~"*100)


