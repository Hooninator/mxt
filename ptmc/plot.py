import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dataclasses import dataclass, asdict

from collections import defaultdict, OrderedDict


import argparse
import os


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   CONFIGS 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass(frozen=True)
class Config:
    tensor:str
    ordering:str
    matrix_gen:str
    norm:str
    precision:str
    rows:str


def parse_configs(path):
    err_configs = {}
    norm_configs = {}
    for file in os.listdir(path):
        stuff = file.split(".csv")[0].split("_")
        tensor = stuff[0]
        if tensor=="randn" or tensor=="evil":
            tensor += f"_{stuff[1]}"
            del stuff[1]
        ordering = stuff[1]
        matrix_gen = stuff[2]
        # Stupid idiot
        norm =""
        j = 0
        for i in range(3, len(stuff)):
            j = i
            if "fp" in stuff[i]:
                norm = norm[:-1]
                break
            norm += stuff[i]+"_"
        precision = stuff[j]
        rows = stuff[j+1]
        kind = stuff[j+2]

        config = Config(tensor, ordering, matrix_gen, norm, precision, rows)

        df = pd.read_csv(f"{path}/{file}")
        if "als_time" not in df.columns:
            df["als_time"] = 0

        if kind=="err":
            err_configs[config] = df
        elif kind=="timing":
            norm_configs[config] = df
        else:
            raise Exception("Bad kind")

    return err_configs, norm_configs


def filter_configs(configs, *args):
    result = {}
    for config in configs:
        d = asdict(config)
        good = True
        for attr, crit in args:
            if d[attr]!=crit:
                good = False
        if good:
            result[config] = configs[config]
    return result


def get_all_uq(configs, attr):
    s = set()
    for config in configs:
        d = asdict(config)
        s.add(d[attr])
    return s


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                 NORM PLOTS 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_norms_tensor(tensor_configs, tensor):
    
    precisions = get_all_uq(tensor_configs, "precision")
    
    for precision in precisions:
        print(f"Making plot for {tensor}_{precision}...")
        plt.figure(figsize=(10, 6))
        plt.grid(True, axis='both', linestyle='-',
                 color='gray', alpha=0.5, zorder=1)
        this_configs = filter_configs(tensor_configs, ("precision", precision))
        df_dict = defaultdict(list)
        for config in this_configs:
            df_dict["c_norms"].append(tensor_configs[config]["c_norm"][0])
            df_dict["f_norms"].append(tensor_configs[config]["f_norm"][0])
            df_dict["matrix_gen"].append(config.matrix_gen)
            df_dict["norm"].append(config.norm)

        df = pd.DataFrame(df_dict)
        df.fillna(1, inplace=True)

        if not os.path.isdir(f"./plots/norms/{tensor}"):
            os.mkdir(f"./plots/norms/{tensor}")

        width = 0.1
        offset = -width * 3

        color_map =  {
                "kronecker_diag_infnorm": "seagreen",
                "null": "crimson",
                "once_two_sided": "salmon",
                "two_sided": "steelblue",
                "one_sided": "navy",
                "kronecker_diag_normal": "lime",
                "kronecker_diag_als": "bisque",
                "kronecker_diag_once": "firebrick"}

        for norm in df["norm"].unique():
            df_this = df[df["norm"]==norm].sort_values(by="matrix_gen")
            x = np.arange(len(df_this["matrix_gen"]))
            y = df_this["c_norms"]
            plt.bar(x + offset, y, label=norm, edgecolor='black', zorder=2, width=width, color=color_map[norm])
            offset += width
        plt.yscale("log")
        plt.ylabel("Error")
        plt.xlabel("Matrix Generation Strategy")
        plt.xticks(x, labels=df_this["matrix_gen"])
        plt.title(f"Componentwise Error for {tensor} in {precision}")
        plt.legend()
        plt.savefig(f"./plots/norms/{tensor}/{tensor}_{precision}_cnorm")
        plt.clf()

        plt.figure(figsize=(10, 6))
        plt.grid(True, axis='both', linestyle='-',
                 color='gray', alpha=0.5, zorder=1)
        width = 0.1
        offset = -width * 3
        for norm in df["norm"].unique():
            df_this = df[df["norm"]==norm].sort_values(by="matrix_gen")
            x = np.arange(len(df_this["matrix_gen"]))
            y = df_this["f_norms"]
            plt.bar(x + offset, y, label=norm, edgecolor='black', zorder=2, width=width, color=color_map[norm])
            offset += width
        plt.yscale("log")
        plt.xlabel("Matrix Generation Strategy")
        plt.ylabel("Error")
        plt.xticks(x, labels=df_this["matrix_gen"])
        plt.title(f"Frobenius Error for {tensor} in {precision}")
        plt.legend()

        plt.savefig(f"./plots/norms/{tensor}/{tensor}_{precision}_fnorm")

        plt.clf()

def plot_norms(dir):

    tensors = os.listdir(dir)

    for tensor in tensors:
        path = f"{dir}/{tensor}"
        tensor_configs, _ = parse_configs(path)
        plot_norms_tensor(tensor_configs, tensor)
        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                 TIMING PLOTS 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_timing_tensor(tensor_configs, tensor):
    
    precisions = get_all_uq(tensor_configs, "precision")
    plt.grid(True, axis='both', linestyle='-',
             color='gray', alpha=0.5, zorder=1)
    
    df_dict = defaultdict(list)
    for precision in precisions:
        this_configs = filter_configs(tensor_configs, ("precision", precision), ("matrix_gen", "uniform"))
        for config in this_configs:
            df_dict["normalization_time"].append((this_configs[config]["tensor_recover_time"] + this_configs[config]["tensor_norm_time"] + this_configs[config]["matrix_norm_time"]).sum())
            df_dict["ttm_time"].append(this_configs[config]["ttm_time"].sum())
            df_dict["ttmc_baseline_time"].append(this_configs[config]["ttmc_baseline_time"].sum())
            df_dict["als_time"].append(this_configs[config]["als_time"].sum())
            df_dict["norm"].append(config.norm)
            df_dict["precision"].append(config.precision)

    df = pd.DataFrame(df_dict)
    df.fillna(1, inplace=True)

    if not os.path.isdir(f"./plots/timing/{tensor}"):
        os.mkdir(f"./plots/timing/{tensor}")

    phases = {"normalization_time", "als_time", "ttm_time"}
    for norm in df["norm"].unique():
        print(f"Making plot for {tensor} {norm}")
        df_this = df[df["norm"]==norm].sort_values(by="precision")
        x = np.arange(len(df_this["precision"]))

        bottom = np.zeros(len(x))
        for phase in phases:
            y = df_this[phase]*1e3
            plt.bar(x, y, label=phase, edgecolor='black', zorder=2, bottom=bottom)
            bottom += y

        plt.ylabel("Runtime (ms)")
        plt.xlabel("Compute Precision")
        plt.xticks(x, labels=df_this["precision"])
        plt.title(f"Runtime Breakdown for {tensor} using Normalization {norm}")
        plt.legend()
        plt.savefig(f"./plots/timing/{tensor}/{tensor}_{norm}_timing")
        plt.clf()


def plot_timing(dir):

    tensors = os.listdir(dir)

    for tensor in tensors:
        path = f"{dir}/{tensor}"
        _, tensor_configs = parse_configs(path)
        plot_timing_tensor(tensor_configs, tensor)
        


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                 DRIVERS 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_ordering_tensor(tensor_configs, tensor):

    mat_rows = get_all_uq(tensor_configs, "rows")

    if not os.path.isdir(f"./plots/ordering/{tensor}"):
        os.mkdir(f"./plots/ordering/{tensor}")

    for row in mat_rows:
        print(f"Plotting {tensor} {row}")
        plt.grid(True, axis='both', linestyle='-',
                 color='gray', alpha=0.5, zorder=1)
        df_dict = defaultdict(list)
        this_configs = filter_configs(tensor_configs, ("rows", row), ("ordering", "rand"), ("precision", "fp16"))
        for config in this_configs:
            df_dict[config.matrix_gen] = this_configs[config]["c_norm"]

        df = pd.DataFrame(df_dict)
        df.fillna(1, inplace=True)
        
        x = np.arange(len(df["uniform"]))
        y = df["uniform"]
        plt.plot(x, y, label="uniform", marker='x', color='steelblue')

        plt.yscale("log") 
        plt.ylabel("Error")
        plt.xlabel("Multiplication Ordering")
        plt.title(f"Componentwise Errors for {tensor} with Output Size {row}")
        plt.legend()
        plt.savefig(f"./plots/ordering/{tensor}/{tensor}_{row}_ordering_unif")

        x = np.arange(len(df["big"]))
        y = df["big"]
        plt.plot(x, y, label="big", marker='x', color='crimson')
        plt.legend()

        plt.savefig(f"./plots/ordering/{tensor}/{tensor}_{row}_ordering")

        plt.clf()


def plot_ordering(dir):
    tensors = os.listdir(dir)

    for tensor in tensors:
        path = f"{dir}/{tensor}"
        tensor_configs, _= parse_configs(path)
        plot_ordering_tensor(tensor_configs, tensor)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()

    if "norms" in args.dir:
        plot_timing(args.dir)
        plot_norms(args.dir)
    elif "ordering" in args.dir:
        plot_ordering(args.dir)
