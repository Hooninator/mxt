
import argparse
import numpy as np
import torch
import random

precisions = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64
}



def main():
    parser = argparse.ArgumentParser(description="Generate a random tensor with high dynamic range in FROSTT format.")
    parser.add_argument("--order", type=int, required=True, help="Order (number of modes) of the tensor.")
    parser.add_argument("--mode_sizes", type=int, nargs='+', required=True, help="Sizes of each mode (list of integers).")
    parser.add_argument("--nnz", type=int, required=True, help="Number of nonzero entries.")
    parser.add_argument("--output", type=str, required=True, help="Output filename (.tns format).")
    parser.add_argument("--precision", type=str, required=True, help="Precision")
    parser.add_argument("--overflow", action="store_true")

    args = parser.parse_args()

    smallest = torch.finfo(precisions[args.precision]).eps
    biggest = torch.finfo(precisions[args.precision]).max

    order = args.order
    mode_sizes = args.mode_sizes
    nnz = args.nnz
    filename = args.output

    if len(mode_sizes) != order:
        raise ValueError(f"Expected {order} mode sizes, but got {len(mode_sizes)}.")

    # Generate indices for each mode
    indices = np.zeros((nnz, order), dtype=int)
    for i in range(order):
        indices[:, i] = np.random.randint(low=0, high=mode_sizes[i], size=nnz) + 1  # 1-based indexing


    assert mode_sizes[0] <= nnz

    values = torch.zeros(nnz)
    for j in range((mode_sizes[0])):
        indices[j, :] = 1
        indices[j, 0] = j + 1
        if args.overflow:
            values[j] = biggest * random.randint(1, 10)
        else:
            values[j] = biggest 

    values[(mode_sizes[0]): ] = smallest * torch.randn(nnz - (mode_sizes[0]))


    # Write to file in FROSTT format
    with open(filename, "w") as f:
        f.write(f"{order}\n{' '.join(map(str, mode_sizes))}\n{nnz}\n")
        for idx, val in zip(indices, values):
            index_str = " ".join(map(str, idx))
            f.write(f"{index_str} {val:.6e}\n")

if __name__ == "__main__":
    main()


