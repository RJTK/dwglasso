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
        self.t = 0  # Current time

        # _s is the internal state of the system
        self._s = np.zeros(self.n * self.p)
        if x_0:
            if len(x_0) == self.n:  # Initialize just the output
                self._s = np.zeros(self.n * self.p)
                self._s[:self.n] = x_0
            elif len(x_0) == self.n * self.p:  # Initialize whole state
                self._s = x_0
            else:
                raise ValueError('Dimension %d of x_0 is not compatible with '
                                 'system dimensions n = %d, p = %d' % (self.n,
                                                                       self.p))
        self.x = self._s[:self.n]  # System output
        return
