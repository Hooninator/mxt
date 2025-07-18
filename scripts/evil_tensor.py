
import argparse
import numpy as np
import torch
import random

from utils import *

precisions = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64
}



def main():
    parser = argparse.ArgumentParser(description="Generate a random tensor with high dynamic range in FROSTT format.")
    parser.add_argument("--order", type=int, required=True, help="Order (number of modes) of the tensor.")
    parser.add_argument("--mode_sizes", type=int, nargs='+', required=True, help="Sizes of each mode (list of integers).")
    parser.add_argument("--output", type=str, required=True, help="Output filename (.tns format).")
    parser.add_argument("--precision", type=str, required=True, help="Precision")

    args = parser.parse_args()

    smallest = torch.finfo(precisions[args.precision]).eps
    biggest = torch.finfo(precisions[args.precision]).max

    order = args.order
    mode_sizes = args.mode_sizes
    filename = args.output

    if len(mode_sizes) != order:
        raise ValueError(f"Expected {order} mode sizes, but got {len(mode_sizes)}.")

    S = torch.randn(size=args.mode_sizes)
    values = torch.full(args.mode_sizes, smallest)
    for i in range(mode_sizes[0]):
        values[i] = biggest
    values = torch.mul(S, values)


    write_dns(filename, values)

if __name__ == "__main__":
    main()


