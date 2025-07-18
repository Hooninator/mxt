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


def write_dns(filename, X):
    with open(filename, 'w') as file:
        vals = X.flatten()
        file.write(f"{len(X.shape)}\n{' '.join([str(s) for s in X.shape])}\n{vals.size()[0]}\n")
        for i in range(len(vals)):
            if i % 10000==0:
                print(f"Writing {i}/{len(vals)}")
            file.write(f"{vals[i]}\n")

