from utils import *
import tensorly as tl


X = torch.tensor(tl.datasets.load_indian_pines().tensor)
filename = "../tensors/indian_pines.dns"
write_dns(filename, X)
