'''VAR(p) model'''
import numpy as np


class VAR(object):
    '''VAR(p) system'''
    def __init__(self, B, x_0=None):
        '''Initializes the model with a list of coefficient matrices
        B = [B(1), B(2), ..., B(p)] where each B(\tau) \in \R^{n \times n}

        x_0 can serve to initialize the system output (if len(x_0) == n)
        or the entire system state (if len(x_0) == n * p)
        '''
        self.p = len(B)
        self.n = (B[0].shape[0])
        if not all(
                len(B_tau.shape) == 2 and  # Check B(\tau) is a matrix
                B_tau.shape[0] == self.n and  # Check compatible sizes
                B_tau.shape[1] == self.n  # Check square
                for B_tau in B):
            raise ValueError('Coefficients must be square matrices of'
                             'equivalent sizes')
        self.B = B  # Keep the list of matrices
        self._B = np.vstack(B)  # Standard form layout \hat{x(t)} = B^\T z(t)
        self.t = 0

        # _z is the internal state of the system
        self._z = np.zeros(self.n * self.p)
        self.reset(x_0=x_0)  # Reset system state
        return

    def induced_graph(self):
        '''Returns the adjacency matrix of the Granger-causality graph
        induced by this VAR model.  This graph is defined via:

        G_{ij} = 1 if \exists tau s.t. B(tau)_{ji} \ne 0, and G_{ij} = 0
        otherwise.

        The interpretation of B(tau)_{ji} is as the coefficient transferring
        energy from process i to process j with a lag of tau seconds.  And,
        G_{ij} = 1 if there is some transfer from process i to j.  Hence,
        we are looking at transposes of coefficient matrices to get to
        the graph adjacency matrix.
        '''
        # We are careful to return np arrays with float type as
        # True/False do not always behave in the same way as 1./0.
        return np.array(sum(B_tau != 0 for B_tau in self.B) != 0,
                        dtype=np.float64)

    def is_stable(self, margin=1e-6):
        '''Checks whether or not the system is stable.  In order to do
        this we directly calculate the eigenvalues of the block companion
        matrix induced by B, which is of size (np x np).  This may be
        prohibitive for very large systems.

        Stability is determined by the spectral radius of the matrix:

        C =
        [B0, B1, B2, ... Bp-1]
        [ I,  0,  0, ... 0   ]
        [ 0,  I,  0, ... 0   ]
        [ 0,  0,  I, ... 0   ]
        [ 0,  0, ..., I, 0   ]

        we return True if |\lambda_max(C)| <= 1 - margin.  Note that the
        default margin is very small.
        '''
        n, p = self.n, self.p
        C = np.hstack((np.eye(n * (p - 1)),  # The block diagonal I
                       np.zeros((n * (p - 1), n))))  # right col
        C = np.vstack((np.hstack((B_tau for B_tau in self.B)),  # top row
                       C))
        ev = np.linalg.eigvals(C)  # Compute the eigenvalues
        return max(abs(ev)) <= 1 - margin

    def reset(self, x_0=None, reset_t=False):
        '''Reset the system to some initial state.  If x_0 is specified,
        it may be of dimension n or n * p.  If it is dimension n, we simply
        dictate the value of the current output, otherwise we reset the
        whole system state.  If reset_t is True then we set the current
        time to reset_t'''
        n, p = self.n, self.p
        if x_0:
            if len(x_0) == n:  # Initialize just the output
                self._z = np.zeros(n * p)
                self._z[:n] = x_0
            elif len(x_0) == n * p:  # Initialize whole state
                self._z = x_0
            else:
                raise ValueError('Dimension %d of x_0 is not compatible with '
                                 'system dimensions n = %d, p = %d' % (n, p))

        else:
            self._z = np.zeros(n * p)
        self.x = self._z[:n]  # System output
        if reset_t:
            self.t = 0
        return

    def drive(self, u):
        '''
        Drives the system with input u.  u should be a T x n array of
        containing a sequence of T inputs, or a single length n input.
        '''
        n, p = self.n, self.p
        if len(u.shape) == 1:  # A single input
            try:
                u = u.reshape((1, n))  # Turn it into a vector
            except ValueError:
                raise ValueError('The length %d of u is not compatible with '
                                 'system dimensions n = %d, p = %d'
                                 % (len(u), n, p))

        if u.shape[1] != n:  # Check dimensions are compatible
            raise ValueError('The dimension %d of the input vectors is '
                             'not compatible with system dimensions n = %d, '
                             ' p = %d' % (u.shape[1], n, p))

        T = u.shape[0]  # The number of time steps
        self.t += T

        # Output matrix to be returned
        Y = np.empty((T, n))
        for t in range(T):
            y = np.dot(self._z, self._B) + u[t, :]  # Next output
            Y[t, :] = y
            self._z = np.roll(self._z, n)  # Update system state
            self._z[:n] = y
        self.x = self._z[:n]  # System output
        return Y
