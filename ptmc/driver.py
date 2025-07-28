from yaml import load, Loader

import subprocess
import argparse

from collections import defaultdict

from dataclasses import dataclass

from ttmc import *


def build_command(tensor, matrix_gen, matrix_row, normalizer, precision, ordering):
    return f"python3 ttmc.py --tensor {tensor} --ordering {ordering} --mat_rows {' '.join(map(str, matrix_row))} --mat_init {matrix_gen} --norm {normalizer} --accum {precision} --compute {precision} --out fp64 --compute_err"

@dataclass
class Config:
    tensor:str
    ordering:str
    ntrials:int
    mat_rows:list[int]
    mat_init:str
    norm:str
    accum:str
    compute:str
    out:str
    profile: bool
    dir: str


def build_config(tensor, matrix_gen, matrix_row, normalizer, precision, ordering, dir):
    config = Config(tensor, ordering, 10, matrix_row, matrix_gen, normalizer, precision, precision, "fp64", False, dir)
    return config


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()

    with open(args.file, 'r') as file, open ("log.out", 'w') as logfile:
        yaml_txt = file.read()
        configs = load(yaml_txt, Loader=Loader)

        tensors = configs["tensors"]
        matrix_gens = configs["matrix_gens"]
        matrix_rows_all = configs["matrix_rows"]
        normalizers = configs["normalizers"]
        precisions = configs["precisions"]
        orderings = configs["orderings"]

        for tensor in tensors:
            matrix_rows = matrix_rows_all[tensor]
            print(f"Reading {tensor}")
            X = read_tensor(tensor, "fp64")
            print("Done")
            for matrix_gen in matrix_gens:
                for matrix_row in matrix_rows:
                    for normalizer in normalizers:
                        for precision in precisions:
                            for ordering in orderings:

                                config = build_config(tensor, matrix_gen, matrix_row, normalizer, precision, ordering, args.dir)

                                filestr = f"{tensor}_{ordering}_{matrix_gen}_{normalizer}_{precision}_{'x'.join(map(str, matrix_row))}"

                                if os.path.exists(f"./data_{args.dir}/{tensor}/{filestr}_timing.csv"):
                                    print(f"{config} already run")
                                    continue

                                logfile.write(f"Running {config}\n...")
                                print(f"Running {config}\n...")
                                
                                tl.set_backend('pytorch')
                                set_torch_flags(config)
                                U_list = init_matrices(config, config.mat_rows, X.shape)

                                try:
                                    main(X, U_list, config)
                                    logfile.write(f"Success:{config}")
                                except Exception as e:
                                    print(f"Command failed -- {e}")
                                    logfile.write("Command failed -- stderr:\n")

