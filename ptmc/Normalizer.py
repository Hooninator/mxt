import torch
import numpy as np
import tensorly as tl
import recover_kron_norm 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#            UTILITY FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def diag_scale_tensor_left(X, D):
    return

def diag_scale_tensor_right(X, D):
    return

def diag_scale_matrix_left(U, D):
    return

def diag_scale_matrix_right(U, D):
    return


def tensor_times_kron(X, matrices, alpha, reciprocal):
    scale = tl.tenalg.kronecker(matrices)
    scale_v = scale.view(X.shape)
    if reciprocal:
        scale_v.reciprocal_()
    X.mul_(scale_v).mul_(alpha)
    return X



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          STANDARD NORMALIZERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class NormalizerStandard:

    def __init__(self, order, accum, compute, out, theta):
        self.R_mat =None  
        self.S_mat =None  
        self.D_mat =None  
        self.accum = accum
        self.compute = compute
        self.out = out
        self.theta = theta


    def normalize_tensor(self, X, mode):
        return X if X.dtype==self.compute else X.to(self.compute)

    def normalize_matrix(self, U, mode):
        return U if U.dtype==self.compute else U.to(self.compute)

    def recover_tensor(self, X, mode):
        return X if X.dtype==self.out else X.to(self.out)


class NormalizerTwoSided(NormalizerStandard):

    def normalize_tensor(self, X, mode):
        
        dims = X.shape

        # Make S (column norm)
        S = torch.abs(X).amax(dim=mode)
        S[S==0] = 1
        S.reciprocal_()

        # Apply S and theta in one step
        scale_shape = [dims[i] if i != mode else 1 for i in range(X.ndim)]
        X.mul_(S.view(scale_shape)).mul_(self.theta)

        # Make R (row norm)
        R = torch.abs(X).amax(dim=[i for i in range(X.ndim) if i != mode])
        R[R==0] = 1
        R.reciprocal_()

        # Apply R in-place
        row_shape = [-1 if i == mode else 1 for i in range(X.ndim)]
        X.mul_(R.view(row_shape))

        self.S_mat = S
        self.R_mat = R
        return X if X.dtype == self.compute else X.to(self.compute)


    def normalize_matrix(self, U, mode):

        # Apply R^-1
        self.R_mat.reciprocal_()
        U.mul_(self.R_mat)

        # Make D
        D = torch.abs(U).amax(dim=1)
        D.reciprocal_()

        # Apply D
        U.mul_(D.unsqueeze(1)).mul_(self.theta)

        self.D_mat = D

        return U if U.dtype==self.compute else U.to(self.compute)


    def recover_tensor(self, X, mode):

        dims = X.shape

        X = X if X.dtype == self.out else X.to(self.out)

        # D^-1
        self.D_mat.reciprocal_().mul_(1/(self.theta*self.theta))
        row_shape = [-1 if i == mode else 1 for i in range(X.ndim)]

        # S^-1
        self.S_mat.reciprocal_()
        col_shape = [dims[i] if i != mode else 1 for i in range(X.ndim)]

        X.mul_(self.D_mat.view(row_shape)).mul_(self.S_mat.view(col_shape))

        #X.mul_(1.0 / (self.theta * self.theta))

        return X


class NormalizerOnceTwoSided(NormalizerTwoSided):

    def normalize_tensor(self, X, mode):
        if mode==0:
            return super().normalize_tensor(X, mode)
        else:
            return X if X.dtype==self.compute else X.to(self.compute)


    def normalize_matrix(self, U, mode):
        if mode==0:
            return super().normalize_matrix(U, mode)
        else:
            # Make D
            D = U.abs().amax(dim=1)
            D.reciprocal_()

            # Apply D
            U = U * D.unsqueeze(1)
            U.mul_(self.theta)

            self.D_mat = D

            return U if U.dtype==self.compute else U.to(self.compute)


    def recover_tensor(self, X, mode):
        if mode==0:
            return super().recover_tensor(X, mode)
        else:
            X = X if X.dtype==self.out else X.to(self.out)
            self.D_mat.reciprocal_()
            X = X * self.D_mat.view([-1 if i==mode else 1 for i in range(X.ndim)])
            X.mul_(1/(self.theta))
            return X


class NormalizerOneSided(NormalizerTwoSided):

    def normalize_tensor(self, X, mode):
        
        dims = X.shape

        # Make S (column norm)
        S = torch.abs(X).amax(dim=mode)
        S[S==0] = 1
        S.reciprocal_()

        # Apply S and theta in one step
        scale_shape = [dims[i] if i != mode else 1 for i in range(X.ndim)]
        X.mul_(S.view(scale_shape)).mul_(self.theta)

        self.S_mat = S
        return X if X.dtype == self.compute else X.to(self.compute)


    def normalize_matrix(self, U, mode):

        # Make D
        D = U.abs().amax(dim=1)
        D.reciprocal_()

        # Apply D
        U.mul_(D.unsqueeze(1)).mul_(self.theta)

        self.D_mat = D

        return U if U.dtype==self.compute else U.to(self.compute)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          KRONECKER NORMALIZERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class KroneckerNormalizer:

    def __init__(self, order, accum_u, compute_u, out_u, theta):
        self.order = order
        self.accum = accum_u
        self.compute = compute_u
        self.out = out_u
        self.theta = theta
        self.D_mats = []
        self.R_mats = []


class KroneckerNormalizerDiag(KroneckerNormalizer):

    def normalize_tensor(self, X):
        X = tensor_times_kron(X, self.D_mats, self.theta, False)
        return X if X.dtype==self.compute else X.to(self.compute)


    def normalize_matrices(self, U_list):
        N = len(U_list)
        return [self.normalize_matrix(U_list[n], n) for n in range(N)]


    def normalize_matrix(self, U, mode):

        self.D_mats[mode].reciprocal_()
        U.mul_(self.D_mats[mode])

        R = U.abs().amax(dim=1)
        R.reciprocal_()
        U.mul_(R.unsqueeze(1)).mul_(self.theta)

        return U if U.dtype==self.compute else U.to(self.compute)


    def recover_tensor(self, X):
        X = recover_kron_norm.diag_unscale_forward2(X, self.R_mats, 1/(self.theta**(X.ndim + 1)), True, self.out)
        #X = X if X.dtype==self.out else X.to(self.out)
        #for mode, d in enumerate(self.R_mats):
        #    # Reshape d to be broadcastable to X
        #    shape = [1] * X.ndim
        #    shape[mode] = -1
        #    if mode==0:
        #        X = X * (d.view(shape) * (1/self.theta**(X.ndim+1)))
        #    else:
        #        X = X * d.view(shape) 
        return X 


    def reset(self):
        self.D_mats = []
        self.R_mats = []


class NormalizerKroneckerDiagNormal(KroneckerNormalizerDiag):

    def init_matrices(self, X, maxiters):
        N = X.ndim
        for i in range(N):
            D = torch.rand(X.shape[i], device='cuda:0', dtype=X.dtype)
            self.D_mats.append(D)


class NormalizerKroneckerDiagInfNorm(KroneckerNormalizerDiag):

    def init_matrices(self, X, maxiters):
        N = X.ndim
        for i in range(N):
            D = torch.abs(X).amax(dim=[n for n in range(N) if n != i]).to(device='cuda:0', dtype=X.dtype)
            D[D==0] = 1
            D.reciprocal_()
            row_shape = [-1 if j == i else 1 for j in range(X.ndim)]
            X.mul_(D.view(row_shape))
            self.D_mats.append(D)
        X = tensor_times_kron(X, self.D_mats, 1, True)


class NormalizerKroneckerDiagALS(KroneckerNormalizerDiag):

    def init_matrices(self, X, maxiters):

        N = X.ndim

        for i in range(N):
            D = torch.rand(X.shape[i], device='cuda:0', dtype=X.dtype)
            self.D_mats.append(D)

        for iter in range(maxiters):

            # Init gamma vectors
            gamma_vecs = []
            for n in range(N):
                gamma = torch.randn(X.shape[n], device='cuda:0', dtype=X.dtype)
                gamma_vecs.append(gamma)

            # Single ALS Sweep
            for n in range(N):
                In = X.shape[n]

                scale = tl.tenalg.kronecker(self.D_mats, skip_matrix=n)
                scale_v = scale.view([X.shape[i] if i != n else 1 for i in range(N)])

                X.mul_(scale_v)
                
                self.D_mats[n] = tl.tenalg.multi_mode_dot(X, gamma_vecs, skip=n)

                s = torch.linalg.vector_norm(X, dim=tuple([d for d in range(N) if d != n]))
                s.reciprocal_()
                self.D_mats[n].mul_(s)

                scale_v.reciprocal_()
                X.mul_(scale_v)


class NormalizerKroneckerDiagOnce(KroneckerNormalizerDiag):

    def init_matrices(self, X, maxiters):
        N = X.ndim
        D = torch.abs(X).amax(dim=[n for n in range(N) if n != 0]).to(device='cuda:0', dtype=X.dtype)
        D[D==0] = 1
        D.reciprocal_()
        self.D_mats.append(D)


    def normalize_tensor(self, X):
        row_shape = [-1 if i == 0 else 1 for i in range(X.ndim)]
        X.mul_(self.D_mats[0].view(row_shape)).mul_(self.theta)
        return X if X.dtype==self.compute else X.to(self.compute)


    def normalize_matrix(self, U, mode):

        if mode==0:
            self.D_mats[mode].reciprocal_()
            U.mul_(self.D_mats[mode])

        R = torch.abs(U).amax(dim=1)
        R.reciprocal_()
        U.mul_(R.unsqueeze(1)).mul_(self.theta)
        R.reciprocal_()

        self.R_mats.append(R)

        return U if U.dtype==self.compute else U.to(self.compute)



class KroneckerNormalizerGeneral(KroneckerNormalizer):

    def normalize_tensor(self, X):
        return X if X.dtype==self.compute else X.to(self.compute)


    def normalize_matrix(self, U, mode):
        return U if U.dtype==self.compute else U.to(self.compute)


    def recover_tensor(self, X):
        X = X if X.dtype==self.out else X.to(self.out)
        return X


    def reset(self):
        self.D_mats = []
        self.R_mats = []


class NormalizerKroneckerGeneralALS(KroneckerNormalizerGeneral):

    def init_matrices(self, X, maxiters):
        N = X.ndim


def make_normalizer(norm, order, accum_u, compute_u, out_u, theta):
    if norm=="null":
        return NormalizerStandard( order,accum_u, compute_u, out_u, theta)
    elif norm=="two_sided":
        return NormalizerTwoSided( order,accum_u, compute_u, out_u, theta)
    elif norm=="once_two_sided":
        return NormalizerOnceTwoSided( order,accum_u, compute_u, out_u, theta)
    elif norm=="one_sided":
        return NormalizerOneSided( order,accum_u, compute_u, out_u, theta)
    elif norm=="kronecker_diag_normal":
        return NormalizerKroneckerDiagNormal( order,accum_u, compute_u, out_u, theta)
    elif norm=="kronecker_diag_als":
        return NormalizerKroneckerDiagALS( order,accum_u, compute_u, out_u, theta)
    elif norm=="kronecker_diag_infnorm":
        return NormalizerKroneckerDiagInfNorm( order,accum_u, compute_u, out_u, theta)
    elif norm=="kronecker_diag_once":
        return NormalizerKroneckerDiagOnce( order,accum_u, compute_u, out_u, theta)
