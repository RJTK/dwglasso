# -*- coding: utf-8 -*-

"""Main module."""
import sys
import numpy as np
import numba
import warnings
import enum


from scipy.linalg import lu_solve, lu_factor
from sklearn.base import BaseEstimator, RegressorMixin


class FitMethod(enum.Enum):
    OLS = 0
    OLST = 1
    LASSO = 2
    DWGLASSO = 3


class DWGLASSO(BaseEstimator, RegressorMixin):
    def __init__(self, p, alpha=0.1, mu=0.1, lmbda=1,
                 tol_abs=0, tol_rel=1e-4, max_iters=100):
        '''
        ADMM implementation (consensus form) used to solve
        the DWGLASSO convex program.

        minimize_B (1 / 2T) * ||Y - ZB||_F^2 +
         lmbda * (alpha * (1 / 2) * ||B||_F^2 + (1 - alpha) * Gamma_DW(B))

        Where Gamma_DW(B) = sum_{ij} ~B~_ij, ~B~_ij = [B_ij(1), ..., B_ij(p)].

        => minimize_B f(B) + g(B)

        f(B) = (1 / 2T) * ||Y - ZB||_F^2 + alpha * lmbda * ||B||_F^2
        g(B) = lmbda * (1 - alpha) * Gamma_DW(B)

        mu is a parameter for the proximal operator:
        prox_psi(v) = argmin_x (psi(x) + (1 / 2mu) * ||x - v||_2^2)

        if lmbda == 0 we are doing Ordinary Least Squares (OLS)
        if lmbda > 0 and alpha == 1 it is tikhonov regularized (OLST)
        '''
        if p <= 0:
            raise ValueError('We require p >= 0')
        self.p = p  # Model order

        if alpha < 0 or alpha > 1:
            raise ValueError('We require alpha \in [0, 1]')
        self.alpha = alpha  # Elastic net F-norm term

        if lmbda < 0:
            raise ValueError('We require lmbda >= 0')
        self.lmbda = lmbda  # Main regularization parameter

        if mu < 0:
            raise ValueError('We require mu >= 0')
        elif (mu == 0 and lmbda > 0 and alpha != 1):
            raise ValueError('We require mu > 0, unless lmbda = 0 (OLS) '
                             'or alpha = 1 when lmbda > 0 (OLST)')
        self.mu = mu  # prox parameter (1/2mu)

        if tol_abs or tol_rel <= 0:
            raise ValueError('We require tolerances to be > 0')
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel

        if max_iters <= 0:
            raise ValueError('We require max_iters > 0')
        self.max_iters = int(max_iters)

        if lmbda == 0:  # OLS
            self.fit_method = FitMethod.OLS
        elif alpha == 1:  # OLST
            self.fit_method = FitMethod.OLST
        elif alpha == 0:  # LASSO
            self.fit_method = FitMethod.LASSO
        else:
            self.fit_method = FitMethod.DWGLASSO

        self.fitted = False  # Indicator if model is fit or not

        self._T = None  # Number of time steps
        self._n = None  # Number of processes
        self._ZTy = None  # Covariance
        self._ZTZ = None  # Variance
        self._lu_piv = None  # LU factorization of ZTZ
        self.coeff_ = None  # The coefficient matrix

        # Diagnostics
        self._err_progress = []
        self._total_iters = 0
        return

    def _proxf(self, V):
        '''Proximity operator of f'''
        return lu_solve(self._lu_piv, self._ZTy + V / self.mu)

    # Create a jitted function
    def _proxg(self):
        '''Proximity operator of g'''
        n = self._n
        p = self.p
        alpha = self.alpha
        mu = self.mu
        lmbda = self.lmbda

        @numba.jit(nopython=True, cache=True)
        def proxg(V):
            P = np.empty((n * p, n))
            for i in range(n):
                for j in range(n):
                    Vtij = V[i::n, j]
                    Vtij_l2 = 0
                    for tau in range(p):
                        Vtij_l2 += Vtij[tau]**2
                    if Vtij_l2 == 0:
                        P[i::n, j] = 0
                    else:
                        r = lmbda * (1 - alpha) * mu / Vtij_l2
                        P[i::n, j] = max(0, 1 - r) * Vtij
            return P
        return proxg

    def _fit_ols(self, silent=True):
        '''Ordinary Least Squares'''
        self._lu_piv = lu_factor(self.ZTZ)
        return lu_solve(self._lu_piv, self.ZTy)

    def _fit_olst(self, silent=True):
        '''Tikhonov regularized Least Squares (ridge regression)'''
        self._lu_piv = lu_factor(self.ZTZ +
                                 self.lmbda * np.eye(self.n * self.p))
        return lu_solve(self._lu_piv, self.ZTy)

    def _fit_lasso(self, silent=True):
        # A specialized method isn't necessary.  Although I could
        # use the sklearn or SPAMS implementation, which is probably faster
        return self._fit_dwglasso(silent)

    def _check_convergence(self, Bu, Bx, Bz, Bz_prev, silent=True):
        '''Checks the residual values against the tolerances and
        returns true if we meet the convergence criteria'''
        # Residual values
        prim_res = np.linalg.norm(Bx - Bz)
        dual_res = np.linalg.norm(Bz_prev - Bz)

        if not silent:
            print('(prim_res, dual_res) = (%0.3f, %0.3f)'
                  % (prim_res, dual_res), end='\r')

        self._prim_err_progress.append(prim_res)
        self._dual_err_progress.append(dual_res)

        prim_tol = self.tol_abs + self.tol_rel * max(np.linalg.norm(Bx),
                                                     np.linalg.norm(Bx))
        dual_tol = self.tol_abs + self.tol_rel * np.linalg.norm(Bu)

        return prim_res <= prim_tol and dual_res <= dual_tol

    def _fit_dwglasso(self, silent=True):
        # Initialize error tracking for convergence diagnostics
        self._prim_err_progress = []
        self._dual_err_progress = []

        # Intermediate variables
        r = (1 + self.mu * self.lmbda * self.alpha) / self.mu
        R = (self._ZTZ + r * np.eye(self._n * self.p))

        # LU factorization for solving a linear equation
        self._lu_piv = lu_factor(R)

        # Initialize a jitted function
        proxg = self._proxg()

        # ADMM iterations
        Bz_prev = np.zeros((self._n * self.p, self._n))
        Bz = np.zeros((self._n * self.p, self._n))
        Bx = self._proxf(Bz)
        Bu = Bx - Bz
        k = 0

        while not self._check_convergence(
                Bu, Bx, Bz, Bz_prev, silent) and k < self.max_iter:
            k += 1
            Bx = self._proxf(Bz - Bu)
            Bz = proxg(Bx + Bu)
            Bu = Bu + Bx - Bz
            rel_err_k = self._rel_err(Bx, Bz)

            if not silent:
                if k == self.max_iter:
                    warnings.warn('Max iterations exceeded!  rel_err = %e'
                                  % rel_err_k, RuntimeWarning)
        if not silent:
            print()
        self._total_iters = k
        return Bz

    def get_B_list(self):
        '''Returns the list of coefficients consumable by VAR()'''
        if not self.fitted:
            raise AssertionError('Model is not fit!')
        return self._B_list

    def fit(self, X, y, silent=True):
        # TODO: Figure out a sensible way to use
        # cached variance / covariance estimates

        # TODO: Allow chaining some other covariance estimator
        # (e.g. graphical LASSO) into DWGLASSO

        self._T = X.shape[0]
        assert len(y) == self._T, 'Incompatible dimensions'

        self._n = int(X.shape[1] // self.p)
        assert X.shape[1] == self._n * self.p, 'Incompatible dimensions'

        # This is a variance matrix of each n processes
        # concatenated with the lagged by 1, 2, ..., p - 1 processes.
        self._ZTZ = (1. / self._T) * np.dot(X.T, X)

        # This is a covariance matrix between x(t) and
        # x(t - 1), ..., x(t - p)
        self._ZTy = (1. / self._T) * np.dot(X.T, y)

        # Fit the model
        if self.fit_method == FitMethod.OLS:
            B = self._fit_ols(silent)
        elif self.fit_method == FitMethod.OLST:
            B = self._fit_olst(silent)
        elif self.fit_method == FitMethod.LASSO:
            B = self._fit_lasso(silent)
        elif self.fit_method == FitMethod.DWGLASSO:
            B = self._fit_dwglasso(silent)
        else:
            raise AssertionError('Non existent fit method')

        self.coeff_ = B
        self._B_list = [B[tau: self.n * (tau + 1), :]
                        for tau in range(self.p)]

        self.fitted = True
        return B

    def transform(self, X, y=None):
        return np.dot(X, self.coeff_)
