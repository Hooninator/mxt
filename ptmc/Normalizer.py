import torch
import numpy as np




class Normalizer:


    def __init__(self, order, accum, compute, out, theta):
        self.R_mat =None  
        self.S_mat =None  
        self.D_mat =None  
        self.accum = accum
        self.compute = compute
        self.out = out
        self.theta = theta


    def normalize_tensor(self, X, mode):
        return X.to(self.compute)

    def normalize_matrix(self, U, mode):
        return U.to(self.compute)

    def recover_tensor(self, X, mode):
        return X.to(self.out)


class NormalizerTwoSided(Normalizer):

    def __init__(self, order, accum, compute, out, theta):
        super().__init__(order, accum, compute, out, theta)


    def normalize_tensor(self, X, mode):

        dims = X.shape

        # Make S
        cols = X.numel() / dims[mode]
        S = X.abs().amax(dim=mode)
        S = torch.reciprocal(S)

        # Apply S
        X = X * S.view([dims[i] if i != mode else 1 for i in range(X.ndim)])
        X.mul_(self.theta)


        # Make R
        R = X.abs().amax(dim=[n for n in range(X.ndim) if n != mode])
        R = torch.reciprocal(R)

        # Apply R
        X =  X * R.view([-1 if i==mode else 1 for i in range(X.ndim)])


        self.R_mat = R
        self.S_mat = S

        return X.to(self.compute)


    def normalize_matrix(self, U, mode):

        # Apply R^-1
        R = torch.reciprocal(self.R_mat)
        U = U * R

        # Make D
        D = U.abs().amax(dim=1)

        # Apply D
        U = U * D.unsqueeze(1)
        U.mul_(self.theta)

        self.D_mat = D

        return U.to(self.compute)


    def recover_tensor(self, X, mode):

        dims = X.shape

        X = X.to(self.out)

        # Apply D^-1
        D = torch.reciprocal(self.D_mat)
        X = X * D.view([-1 if i==mode else 1 for i in range(X.ndim)])

        # Apply S^-1
        S = torch.reciprocal(self.S_mat)
        X = X * S.view([dims[i] if i != mode else 1 for i in range(X.ndim)])

        X.mul_(1/(self.theta * self.theta))

        return X



def make_normalizer(norm, order, accum_u, compute_u, out_u, theta):
    if norm=="null":
        return Normalizer( order,accum_u, compute_u, out_u, theta)
    if norm=="two_sided":
        return NormalizerTwoSided( order,accum_u, compute_u, out_u, theta)

