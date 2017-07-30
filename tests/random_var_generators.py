'''Methods to generate random VAR models'''
import numpy as np


def random_var(graph_func, coefficient_func, margin=1e-6, max_tries=15):
    '''
    Provide functions to generate the graph and the coefficients.  We
    will produce random VAR models from these functions until a stable
    one is found, which is subsequently returned.

    graph_func should take no coefficients, and should return an adjacency
    matrix.  coefficient_func should take as it's argument this adjacency
    matrix and return a list of VAR coefficients.

    To my knowledge, there are no straightforward ways to generate
    "interesting" stable VAR models.  We resort to randomly trying,
    (with some guidence from loose bounds) and then twiddling weights.

    max_tries specifies the maximum number of times we sample a set of
    VAR coefficients before raising a RuntimeError.  Use margin to
    determine the threshold of stability.  See 'is_stable()'
    '''
    k = 0  # Number of attempts
    while k < max_tries:
        G = graph_func()
        B = coefficient_func(G)
        if is_stable(B):
            return B
    raise RuntimeError('Exhausted %d attempts without producing a stable ,'
                       'VAR system.' % max_tries)
    return


def is_stable(B, margin=1e-6):
    '''Checks if B = [B0, ..., Bp] forms a stable VAR(p) system.
    see dwglasso.var.Var, we do much less checking of the inputs here.'''
    p = len(B)
    n = B[0].shape[0]
    C = np.hstack((np.eye(n * (p - 1)),  # The block diagonal I
                   np.zeros((n * (p - 1), n))))  # right col
    C = np.vstack((np.hstack((B_tau for B_tau in B)),  # top row
                   C))
    ev = np.linalg.eigvals(C)  # Compute the eigenvalues
    return max(abs(ev)) <= 1 - margin


def iid_gaussian_var(p, G, r=0.65):
    '''
    Form p matrices of iid gaussians, multiply them all by the same matrix G.
    The gaussian matrices are normalized by

    (pi/2)**.25 * sqrt(r / (n - 1)) / p

    which is a reasonable normalization factor derived from Gershgorin's
    circle theorem.  Tuning r ~= 0.65 / q seems to be a reasonable value to get
    stable systems when the underlying graph has iid Ber(q) edges.

    * Note, we do nothing here to check stability of the system.
    '''
    n = G.shape[0]

    def _dN():  # Gaussian distribution
        return np.random.normal(0, 1, size=(n, n))

    k = ((np.pi / 2)**.25) * np.sqrt(float(r) / (n - 1)) / float(p)
    B = [k * _dN() * G.T for tau in range(p)]
    return B


def iid_ber_graph(n, q):
    '''
    Forms the adjacency matrix of a graph by drawing a BER(p) random variable
    for each of the possible n^2 edges.  Note that we are including self-loops.
    '''
    return np.random.binomial(n=1, p=q, size=(n, n))
