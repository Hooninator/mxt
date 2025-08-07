import tensorly as tl
import numpy as np
import torch
import pandas as pd

import os
import argparse
import random
import time

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
          "ttmc_mixed_time":[],
          "als_time":[]}

errors = {"f_norm":[],
          "c_norm":[]}

precisions = { "fp16":torch.float16,
               "fp32":torch.float32,
               "fp64":torch.float64}


def read_tensor(name, u):
    tensor = []
    with open(f"{base}{name}.dns", 'r') as file:
        nu = 0
        i = 0
        for entry in file:
            nu += 1
            if nu==2:
                shape = list(map(int, entry.split(" ")))
                next(file)
                break
        tensor = [float(s) for line in file for s in line.split()]

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
    if args.mat_init=="uniform":
        return init_matrices_unif(rows, cols, args.out)
    elif args.mat_init=="big":
        return init_matrices_big(rows, cols, args.out)
    elif args.mat_init=="evil":
        return init_matrices_evil(rows, cols, args.out)
    else:
        raise Exception(f"Invalid init {args.mat_init}")


def init_matrices_unif(rows, cols, out):
    matrices = []
    n = len(rows)
    for i in range(n):
        matrices.append(torch.rand(rows[i], cols[i], device='cuda:0', dtype=precisions[out]))
    return matrices


def init_matrices_big(rows, cols, out):
    matrices = []
    n = len(rows)
    biggest = torch.finfo(torch.float16).max
    torch.manual_seed(45)
    torch.cuda.manual_seed(45)
    np.random.seed(45)
    random.seed(45)
    for i in range(n):
        matrices.append(torch.randn(rows[i], cols[i], device='cuda:0', dtype=precisions[out]))
        matrices[i].mul_(biggest)
    return matrices


def init_matrices_evil(rows, cols, out):
    matrices = []
    n = len(rows)
    biggest = torch.finfo(torch.float16).max
    smallest = torch.finfo(torch.float16).eps
    for i in range(n):
        matrices.append(torch.ones(rows[i], cols[i], device='cuda:0', dtype=precisions[out]))
        matrices[i].mul_(smallest)
        for j in range(rows[i]):
            matrices[i][j, 0] = biggest
        matrices[i].mul_(torch.randn(rows[i], cols[i],device='cuda:0', dtype=precisions[out])) 
    return matrices

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                  Ordering
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ordering_smallest_shared(rows, cols):
    N = len(rows)
    ordering = list(range(N))
    return [x for _, x in sorted(zip(rows, ordering), key=lambda p: p[0])]


def ordering_compute(rows, cols):
    N = len(rows)
    pi = np.zeros(N)
    for i in range(N):
        pi[i] = (1 / cols[i]) - (1/rows[i])
    ordering = list(range(N))
    ordering_best = [x for _, x in sorted(zip(pi, ordering), key=lambda p: p[0])]
    return ordering_best


def ordering_rand(rows, cols, t):
    random.seed(t)
    ordering = list(range(len(rows)))
    random.shuffle(ordering)
    return ordering


def ordering_default(rows, cols):
    ordering = list(range(len(rows)))
    return ordering


def make_ordering(ordering, rows, cols, t):
    if ordering=="rand":
        return ordering_rand(rows, cols, t)
    elif ordering=="default":
        return ordering_default(rows, cols)
    elif ordering=="compute":
        return ordering_compute(rows, cols)
    elif ordering=="smallest_shared":
        return ordering_smallest_shared(rows, cols)
    else:
        raise Exception(f"Invalid ordering {ordering}")

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
        torch.cuda.synchronize()
        tensor_norm_time += (time.time() - t1)

        t2 = time.time()
        U = normalizer.normalize_matrix(U_list[mode], mode)
        torch.cuda.synchronize()
        matrix_norm_time += (time.time() - t2)

        # Perform TTM
        t3 = time.time()
        Y = tl.tenalg.mode_dot(Y, U, mode)
        torch.cuda.synchronize()
        ttm_time += (time.time() - t3)

        # Recover output
        t4 = time.time()
        Y = normalizer.recover_tensor(Y, mode)
        torch.cuda.synchronize()
        tensor_recover_time += (time.time() - t4)

    if trial > 0:
        timers["tensor_norm_time"].append(tensor_norm_time)
        timers["matrix_norm_time"].append(matrix_norm_time)
        timers["ttm_time"].append(ttm_time)
        timers["tensor_recover_time"].append(tensor_recover_time)

    return Y


def ttmc_mixed_als(X, U_list, normalizer, ordering, trial):

    tensor_norm_time = 0
    matrix_norm_time = 0
    ttm_time = 0
    tensor_recover_time = 0
    als_time = 0

    N = len(U_list)

    normalizer.reset()

    t1 = time.time()
    normalizer.init_matrices(X, 1)
    torch.cuda.synchronize()
    als_time += (time.time() - t1)

    t1 = time.time()
    X = normalizer.normalize_tensor(X)
    torch.cuda.synchronize()
    tensor_norm_time += (time.time() - t1)
    Y = X

    # Normalize matrices
    t2 = time.time()
    U_list = normalizer.normalize_matrices(U_list)
    torch.cuda.synchronize()
    matrix_norm_time += (time.time() - t2)

    for n in range(N):

        mode = ordering[n]
        U = U_list[mode]

        # Perform TTM
        t3 = time.time()
        Y = tl.tenalg.mode_dot(Y, U, mode)
        torch.cuda.synchronize()
        ttm_time += (time.time() - t3)

    t4 = time.time()
    Y = normalizer.recover_tensor(Y)
    torch.cuda.synchronize()
    tensor_recover_time += (time.time() - t4)

    if trial > 0:
        timers["tensor_norm_time"].append(tensor_norm_time)
        timers["matrix_norm_time"].append(matrix_norm_time)
        timers["ttm_time"].append(ttm_time)
        timers["als_time"].append(als_time)
        timers["tensor_recover_time"].append(tensor_recover_time)

    return Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#              Error Measurements 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def frob_norm_err(Y_correct, Y_computed):
    Y_correct = Y_correct.to(device='cpu')
    Y_computed = Y_computed.to(device='cpu')
    return (torch.norm(Y_correct - Y_computed)) / torch.norm(Y_correct)


def componentwise_err(Y_correct, Y_computed):
    Y_correct = Y_correct.to(device='cpu')
    Y_computed = Y_computed.to(device='cpu')
    err = torch.abs((Y_correct - Y_computed)) 
    ind = torch.unravel_index(torch.argmax(err), Y_correct.shape)
    return err[ind] / torch.abs(Y_correct[ind])




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   Main 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_torch_flags(args):
    if args.accum=="fp16":
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        #torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    if args.accum=="fp32":
        torch.backends.cuda.matmul.allow_tf32 = True


def arg_to_str(args):
    return f"{args.tensor}_{args.ordering}_{args.mat_init}_{args.norm}_{args.compute}_{'x'.join(map(str, args.mat_rows))}"


def print_times():
    print("----TIMES----")
    for timer in timers:
        print(f"\t[{timer}]: {sum(timers[timer])}s")


def fill_empty(d):
    to_del = []
    m = 0
    for k in d:
        m = max(m, len(d[k]))
        if len(d[k])==0:
            to_del.append(k)
    for k in to_del:
        d[k] = [0]*m

    return d


def write_metrics(args):

    global timers
    global errors

    fname = arg_to_str(args)

    if not os.path.isdir(f"./data_{args.dir}/{args.tensor}"):
        os.mkdir(f"./data_{args.dir}/{args.tensor}")
    
    timers = fill_empty(timers)
    errors = fill_empty(errors)

    print(timers)
    timing_df = pd.DataFrame(timers)
    err_df = pd.DataFrame(errors)

    timing_df.to_csv(f"./data_{args.dir}/{args.tensor}/{fname}_timing.csv")
    err_df.to_csv(f"./data_{args.dir}/{args.tensor}/{fname}_err.csv")


def reset_metrics():
    global timers
    global errors

    for k in timers:
        timers[k] = []
    for k in errors:
        errors[k] = []


def main(X, U_list, args):

    # Normalizer
    theta = 0.1
    normalizer = make_normalizer(args.norm, X.ndim, precisions[args.accum], precisions[args.compute], precisions[args.out], theta)

    # Run TTMc
    for t in range(args.ntrials):

        print(f"----TRIAL {t}----")

        X_cpy = X.to(device='cpu')
        U_list_cpy = []
        for i in range(X.ndim):
            U_list_cpy.append(U_list[i].to(device='cpu'))

        ttmc_mixed_time = 0
        ttmc_baseline_time = 0

        # Determine ordering
        ordering = make_ordering(args.ordering, args.mat_rows, X.shape, t)

        t0 = time.time()

        if args.profile:
            with torch.autograd.profiler.profile(use_device='cuda') as prof:
                if "kronecker" in args.norm:
                    Y = ttmc_mixed_als(X, U_list, normalizer, ordering, t)
                else:
                    Y = ttmc_mixed(X, U_list, normalizer, ordering, t)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        else:
            if "kronecker" in args.norm:
                Y = ttmc_mixed_als(X, U_list, normalizer, ordering, t)
            else:
                Y = ttmc_mixed(X, U_list, normalizer, ordering, t)
        torch.cuda.synchronize()
        ttmc_mixed_time = (time.time() - t0)


        X = X_cpy.to(device='cuda:0')
        for i in range(X.ndim):
            U_list[i] = U_list_cpy[i].to(device='cuda:0')

        t9 = time.time()
        Y_correct = tl.tenalg.multi_mode_dot(X, U_list)
        torch.cuda.synchronize()
        ttmc_baseline_time = (time.time() - t9)

        if t > 0:
            timers["ttmc_mixed_time"].append(ttmc_mixed_time)
            timers["ttmc_baseline_time"].append(ttmc_baseline_time)

        c_err = componentwise_err(Y_correct, Y)
        f_err = frob_norm_err(Y_correct, Y)
        errors["f_norm"].append(f_err.item())
        errors["c_norm"].append(c_err.item())
        print(f"\t[frob_norm]: {f_err}")
        print(f"\t[comp_norm]: {c_err}")


    print_times()

    if args.dir:
        write_metrics(args)

    reset_metrics()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", type=str)
    parser.add_argument("--ordering", type=str)
    parser.add_argument("--ntrials", type=int, default=10)
    parser.add_argument("--mat_rows", type=int, nargs='+')
    parser.add_argument("--mat_init", type=str)
    parser.add_argument("--norm", type=str)
    parser.add_argument("--accum", type=str)
    parser.add_argument("--compute", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--dir", type=str)
    parser.add_argument("--profile", action='store_true')

    args = parser.parse_args()

    tl.set_backend('pytorch')
    set_torch_flags(args)

    print("~"*100)
    print(f"Running {args.tensor}")

    # Set up inputs 

    X = read_tensor(args.tensor, "fp64")
    U_list = init_matrices(args, args.mat_rows, X.shape)
    main(X, U_list, args)

    print("~"*100)

