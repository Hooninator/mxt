import numpy as np
import torch

def write_frostt(filename, X):
    inds = np.array(torch.nonzero(X + 1))
    with open(filename, 'w') as file:
        for i in range(inds.shape[0]):
            if i % 1000==0:
                print(f"Writing value {i}/{len(inds)}...")
            indices = inds[i, :] + 1
            line = ' '.join(str(idx) for idx in indices)
            file.write(f"{line} {X[tuple(inds[i, :])]:.18f}\n")


def write_dns_fast(filename, X):
    with open(filename, 'w') as file:
        header=f"{len(X.shape)}\n{' '.join([str(s) for s in X.shape])}\n{torch.numel(X)}"
    np.savetxt(filename, X.cpu().numpy().reshape(-1), fmt="%.10g", header=header, comments='')


def write_dns(filename, X):
    with open(filename, 'w') as file:
        vals = list(X.flatten())
        file.write(f"{len(X.shape)}\n{' '.join([str(s) for s in X.shape])}\n{len(vals)}\n")
        for i in range(len(vals)):
            if i % 10000==0:
                print(f"Writing {i}/{len(vals)}")
            file.write(f"{vals[i]}\n")

