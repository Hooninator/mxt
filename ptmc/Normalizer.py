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

    def diag_scale_tensor_left(X, D):
        return

    def diag_scale_tensor_right(X, D):
        return

    def diag_scale_matrix_left(U, D):
        return

    def diag_scale_matrix_right(U, D):
        return

class NormalizerTwoSided(Normalizer):

    def __init__(self, order, accum, compute, out, theta):
        super().__init__(order, accum, compute, out, theta)


    def normalize_tensor(self, X, mode):
        
        dims = X.shape

        # Make S (column norm)
        S = X.amax(dim=mode).abs_()  # no alloc if dim=mode
        S.reciprocal_()

        # Apply S and theta in one step
        scale_shape = [dims[i] if i != mode else 1 for i in range(X.ndim)]
        X.mul_(S.view(scale_shape)).mul_(self.theta)

        # Make R (row norm)
        R = X.amax(dim=[i for i in range(X.ndim) if i != mode]).abs_()
        R.reciprocal_()

        # Apply R in-place
        row_shape = [-1 if i == mode else 1 for i in range(X.ndim)]
        X.mul_(R.view(row_shape))

        self.S_mat = S
        self.R_mat = R
        return X if X.dtype == self.compute else X.to(self.compute)


    def normalize_matrix(self, U, mode):

        # Apply R^-1
        #R = torch.reciprocal(self.R_mat)
        self.R_mat.reciprocal_()
        U.mul_(self.R_mat)

        # Make D
        D = U.amax(dim=1).abs_()

        # Apply D
        U.mul_(D.unsqueeze(1)).mul_(self.theta)

        self.D_mat = D

        return U if U.dtype==self.compute else U.to(self.compute)


    def recover_tensor(self, X, mode):

        dims = X.shape

        X = X if X.dtype == self.out else X.to(self.out)

        # D^-1
        self.D_mat.reciprocal_()
        row_shape = [-1 if i == mode else 1 for i in range(X.ndim)]
        X.mul_(self.D_mat.view(row_shape))

        # S^-1
        self.S_mat.reciprocal_()
        col_shape = [dims[i] if i != mode else 1 for i in range(X.ndim)]
        X.mul_(self.S_mat.view(col_shape))

        X.mul_(1.0 / (self.theta * self.theta))

        return X


class NormalizerOnceTwoSided(NormalizerTwoSided):

    def __init__(self, order, accum, compute, out, theta):
        super().__init__(order, accum, compute, out, theta)

    def normalize_tensor(self, X, mode):
        if mode==0:
            return super().normalize_tensor(X, mode)
        else:
            return X.to(self.compute)


    def normalize_matrix(self, U, mode):
        if mode==0:
            return super().normalize_matrix(U, mode)
        else:
            # Make D
            D = U.abs().amax(dim=1)

            # Apply D
            U = U * D.unsqueeze(1)
            U.mul_(self.theta)

            self.D_mat = D

            return U.to(self.compute)


    def recover_tensor(self, X, mode):
        if mode==0:
            return super().recover_tensor(X, mode)
        else:
            X = X.to(self.out)
            self.D_mat.reciprocal_()
            X = X * self.D_mat.view([-1 if i==mode else 1 for i in range(X.ndim)])
            X.mul_(1/(self.theta))
            return X


class NormalizerALS:

    def __init__(self, order, accum_u, compute_u, out_u):
        self.order = order
        self.accum_u = accum_u
        self.compute_u = compute_u
        self.out_u = out_u


def make_normalizer(norm, order, accum_u, compute_u, out_u, theta):
    if norm=="null":
        return Normalizer( order,accum_u, compute_u, out_u, theta)
    elif norm=="two_sided":
        return NormalizerTwoSided( order,accum_u, compute_u, out_u, theta)
    elif norm=="once_two_sided":
        return NormalizerOnceTwoSided( order,accum_u, compute_u, out_u, theta)
    elif norm=="als":
        return NormalizerALS( order,accum_u, compute_u, out_u, theta)

