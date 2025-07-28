from utils import *
import tensorly as tl


X = torch.tensor(tl.datasets.load_kinetic().tensor)
filename = "../tensors/kinetic.dns"
write_dns(filename, X)
