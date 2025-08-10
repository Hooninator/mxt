import torch
import recover_kron_norm 

def recover(tY, r_inv, out_dtype=torch.float32, use_fp64_accum=False):
    return recover_kron_norm.diag_unscale_forward(tY, r_inv, use_fp64_accum, out_dtype)
