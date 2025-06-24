import tensorly as tl
import numpy as np
import torch

X = torch.tensor(tl.datasets.load_kinetic().tensor)


inds = torch.nonzero(X)
filename = "kinetic.tns"
with open(filename, 'w') as file:
    for i in range(inds.shape[0]):
        indices = inds[i, :] + 1
        line = ' '.join(str(idx.item()) for idx in indices)
        file.write(f"{line} {X[tuple(inds[i, :])]:.18f}\n")
