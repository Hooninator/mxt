from scipy.io import loadmat

import argparse

from utils import *



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str)
    parser.add_argument("-o", type=str)
    args = parser.parse_args()

    print(f"Loading matrix {args.i}...")
    mat = loadmat(args.i)
    print("Done!")

    k = list(mat.keys())[-1]

    print(f"Writing to {args.o}...")
    write_dns(args.o, mat[k])








