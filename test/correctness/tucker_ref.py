
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import torch
import pandas as pd

from dataclasses import dataclass
from functools import reduce
from collections import defaultdict

from scipy.io import loadmat

import time
import os
import argparse
from yaml import load, Loader


##################################################
#
#               UTILITY FUNCTIONS
#
##################################################

precisions = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64
}

base = "../../tensors/"


def read_tensor(config):
    tensor = torch.zeros(config["shape"], dtype=torch.float64)
    with open(f"{base}{config['name']}.tns", 'r') as file:
        for entry in file:
            stuff = entry.split(" ")[:-1]
            inds = [int(s) - 1 for s in stuff[:-1]]
            tensor[tuple(inds)] = float(stuff[-1])
    return tensor


def compute_error(X, G, U_list):
    T = reconstruct(G, U_list)
    return (torch.linalg.norm(X - T) / torch.linalg.norm(X)).item()


def reconstruct(G, U_list):
    Y = tl.tenalg.multi_mode_dot(G.numpy(), U_list)
    return torch.tensor(Y)


##################################################
#
#               DRIVER AND STATS
#
##################################################


def write_tensor(filename, X):
    inds = np.array(torch.nonzero(X + 1))
    with open(filename, 'w') as file:
        for i in range(inds.shape[0]):
            indices = inds[i, :] + 1
            line = ' '.join(str(idx) for idx in indices)
            file.write(f"{line} {X[tuple(inds[i, :])]:.18f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiters", type=int)
    args = parser.parse_args()

    with open("./test.yaml", 'r') as file:

        yaml_txt = file.read()
        d = load(yaml_txt, Loader=Loader)

        for config_name in d['configs']:
            print("~"*100)
            print(f"Running {config_name}")
            config = d['configs'][config_name]
            
            if not os.path.isdir(f"./cases/{config['name']}"):
                os.mkdir(f"./cases/{config['name']}")

            X = read_tensor(config)

            # Get the factors 
            G_ref, U_list_ref = tl.decomposition.tucker(
                X.numpy(), config["ranks"], n_iter_max=0, tol=1e-16)
            G_ref = torch.tensor(G_ref)

            for i in range(len(U_list_ref)):
                U_list_ref[i] = torch.tensor(U_list_ref[i])
                filename = f"./cases/{config['name']}/factor_{i}.tns"
                write_tensor(filename, U_list_ref[i])

            init_error = compute_error(X, G_ref, U_list_ref)
            print(f"Initial Error: {init_error}")

            # Get the core tensor, run for full iterations 
            G_ref, U_list_ref = tl.decomposition.tucker(
                X.numpy(), config["ranks"], n_iter_max=args.maxiters, tol=1e-16)
            G_ref = torch.tensor(G_ref)
            filename = f"./cases/{config['name']}/core.tns"
            write_tensor(filename, G_ref)

            for i in range(len(U_list_ref)):
                U_list_ref[i] = torch.tensor(U_list_ref[i])

            ref_error = compute_error(X, G_ref, U_list_ref)
            print(f"Reference Error: {ref_error}")

            print("~"*100)

