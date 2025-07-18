import tensorly as tl
import numpy as np
import os
from yaml import load, Loader



base = "../../tensors/"



def read_tensor(config):
    tensor = []
    with open(f"{base}{config['name']}.dns", 'r') as file:
        for entry in file:
            if entry.find(".")==-1:
                continue
            tensor.append(float(entry))
    return np.ndarray(shape=config["shape"], buffer=np.array(tensor), order='C')



def write_tensor(filename, X):
    inds = np.array(np.nonzero(X + 1))
    with open(filename, 'w') as file:
        file.write(f"{len(X.shape)}\n{' '.join(str(idx) for idx in X.shape)}\n{X.size}\n")
        for i in range(inds.shape[1]):
            indices = inds[:, i] + 1
            line = ' '.join(str(idx) for idx in indices)
            file.write(f"{line} {X[tuple(inds[:, i])]:.18f}\n")


def write_dns(filename, X):
    with open(filename, 'w') as file:
        vals = X.flatten(order='C')
        file.write(f"{len(X.shape)}\n{' '.join([str(s) for s in X.shape])}\n{vals.size}\n")
        for i in range(len(vals)):
            if i % 10000==0:
                print(f"Writing {i}/{len(vals)}")
            file.write(f"{vals[i]:.18f}\n")



def init_matrices(rows, cols):
    matrices = []
    n = len(rows)
    for i in range(n):
        matrices.append(np.random.uniform(size=(rows[i], cols[i])))
    return matrices


if __name__ == "__main__":

    with open("./test.yaml", 'r') as file:

        yaml_txt = file.read()
        d = load(yaml_txt, Loader=Loader)

        for config_name in d['configs']:
            print("~"*100)
            print(f"Running {config_name}")
            config = d['configs'][config_name]
            
            if not os.path.isdir(f"./ttmc_golden/{config['name']}"):
                os.mkdir(f"./ttmc_golden/{config['name']}")

            X = read_tensor(config)
            print(X)

            U_list = init_matrices(config['ranks'], config['shape'])
            for i in range(len(U_list)):
                print(U_list[i])
                write_dns(f"./ttmc_golden/{config['name']}/matrix_{i}.dns", U_list[i])

            Y = tl.tenalg.multi_mode_dot(X, U_list)
            write_dns(f"./ttmc_golden/{config['name']}/output.dns", Y)

            print("~"*100)


