from utils import *
import argparse




if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate a random dense Gaussian tensor in FROSTT format.")
    parser.add_argument("--order", type=int, required=True, help="Order (number of modes) of the tensor.")
    parser.add_argument("--mode_sizes", type=int, nargs='+', required=True, help="Sizes of each mode (list of integers).")
    parser.add_argument("--output", type=str, required=True, help="Output filename (.dns format).")
    args = parser.parse_args()

    
    print(f"Generating tensor...")
    vals = torch.randn(size=(args.mode_sizes))
    print(f"Done")
    write_dns_fast(args.output, vals)

