import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate a random Gaussian tensor in FROSTT format.")
    parser.add_argument("--order", type=int, required=True, help="Order (number of modes) of the tensor.")
    parser.add_argument("--mode_sizes", type=int, nargs='+', required=True, help="Sizes of each mode (list of integers).")
    parser.add_argument("--nnz", type=int, required=True, help="Number of nonzero entries.")
    parser.add_argument("--output", type=str, required=True, help="Output filename (.tns format).")
    parser.add_argument("--scaling", type=float, default=1.0)

    args = parser.parse_args()

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

    # Generate Gaussian values multiplied by scaling factor
    values = np.random.randn(nnz) * args.scaling

    # Write to file in FROSTT format
    with open(filename, "w") as f:
        for idx, val in zip(indices, values):
            index_str = " ".join(map(str, idx))
            f.write(f"{index_str} {val:.6e}\n")

if __name__ == "__main__":
    main()


