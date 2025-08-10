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
        if "smallest_shared" in file:
            k = 3
            ordering = "smallest_shared"
        else:
            k = 2
            ordering = stuff[1]

        matrix_gen = stuff[k]

        # Stupid idiot
        norm =""
        j = 0
        for i in range(k+1, len(stuff)):
            j = i
            if "fp" in stuff[i]:
                norm = norm[:-1]
                break
            norm += stuff[i]+"_"
        precision = stuff[j]
        rows = stuff[j+1]
        kind = stuff[j+2]

        config = Config(tensor, ordering, matrix_gen, norm, precision, rows)
        print(config)

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
        plot_timing_tensor2(tensor_configs, tensor)
        

def plot_timing_tensor2(tensor_configs, tensor):
    mat_rows = get_all_uq(tensor_configs, "rows")
    if not os.path.isdir(f"./plots/timing/{tensor}"):
        os.mkdir(f"./plots/timing/{tensor}")

    for row in mat_rows:
        print(f"Plotting {tensor} {row}")
        plt.grid(True, axis='both', linestyle='-',
                 color='gray', alpha=0.5, zorder=1)
        df_dict = defaultdict(list)

        fp16_config_one_sided = list(filter_configs(tensor_configs, ("rows", row), ("precision", "fp16"), ("matrix_gen", "uniform"), ("norm", "one_sided")).keys())[0]
        fp16_config_kronecker_once = list(filter_configs(tensor_configs, ("rows", row), ("precision", "fp16"), ("matrix_gen", "uniform"), ("norm", "kronecker_diag_once")).keys())[0]
        fp16_config_kronecker_inf = list(filter_configs(tensor_configs, ("rows", row), ("precision", "fp16"), ("matrix_gen", "uniform"), ("norm", "kronecker_diag_infnorm")).keys())[0]
        fp64_config = list(filter_configs(tensor_configs, ("rows", row), ("precision", "fp64"), ("matrix_gen", "uniform"), ("norm", "null")).keys())[0]

        phases = {"tensor_norm_time", "matrix_norm_time", "tensor_recover_time", "als_time", "ttm_time"}

        x_labels = ["fp64", "fp16_one_sided", "fp16_kron_once", "fp16_kron_inf"]
        x = np.arange(len(x_labels))
        bottom = np.zeros(len(x_labels))
        for phase in phases:
            y = np.array([tensor_configs[fp64_config][phase].sum(), 
                 tensor_configs[fp16_config_one_sided][phase].sum(),
                 tensor_configs[fp16_config_kronecker_once][phase].sum(),
                 tensor_configs[fp16_config_kronecker_inf][phase].sum()]) * 1e3
            if phase=="als_time":
                phase = "init_time"
            plt.bar(x, y, label=phase, edgecolor='black', zorder=2, bottom=bottom)
            bottom += y

        plt.ylabel("Runtime (ms)")
        plt.xticks(x, labels=x_labels)
        plt.title(f"Runtime Breakdown for {tensor} with Output Size {row}")
        plt.legend()
        plt.savefig(f"./plots/timing/{tensor}/{tensor}_{row}_timing")
        plt.clf()
        



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
            df_dict[config.matrix_gen] = this_configs[config]["f_norm"]

        df = pd.DataFrame(df_dict)
        df.fillna(1, inplace=True)
        
        x = np.arange(len(df["uniform"]))
        y = df["uniform"]
        plt.plot(x, y, label="random", marker='x', color='steelblue')

        this_configs = list(filter_configs(tensor_configs, ("rows", row), ("ordering", "smallest_shared"), ("precision", "fp16"), ("matrix_gen", "uniform")))[0]
        print(this_configs)
        y = tensor_configs[this_configs]["f_norm"]
        plt.plot(x, y, label="smallest_shared", linestyle="--", color='limegreen')
        plt.title(f"Forbenius Norm Error for {tensor} -- Output Size {row}")
        plt.xlabel("Trial")
        plt.ylabel("Error")
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
