from collections import OrderedDict
import copy
from functools import partial
from pprint import pprint
import numpy as np
import sympy
from .numerics import *
from .stack import *

class Gru(object):
    def __init__(self, hidden_size):
        self.N = N = hidden_size
        assert N % 6 == 0
        K = N / 3

        self.U = np.random.randn(2 * K, N)
        self.V = np.random.randn(2 * K, N)

        self.W = np.random.randn(K, N)
        self.G = np.random.randn(K, K)
        self.P = np.random.randn(K, N)

        # only need to invert weights once at start of each backwards pass
        self._A_inv = np.linalg.inv(np.vstack([self.U, self.W]))

        # initial state
        self.h = [np.random.randn(N / 6) for _ in range(6)]

        self._solver = self._build_solver()

    def forward(self, x):
        assert x.size == self.N

        h_tm1 = np.concatenate(self.h)
        z, r = np.split(
            sigmoid(np.dot(self.U, x) + np.dot(self.V, h_tm1)),
            2
        )
        q = np.tanh(
            np.dot(self.W, x) +
            np.dot(self.G, r * np.dot(self.P, h_tm1))
        )

        print '-' * 20
        print z
        print r
        print q
        print '-' * 20

        z, r, q = [np.split(arr, 2) for arr in [z, r, q]]

        # first the GRUs
        off = 0
        for i in range(2):
            self.h[i + off] = z[i] * self.h[i + off] + (1. - z[i]) * q[i]

        # GILRs
        # NOTE: could replace r[0], r[1] with revnet output
        off = 2
        for i in range(2):
            self.h[i + off] = z[i] * self.h[i + off] + r[i]

        # X-GILRs
        off = 4
        for i in range(2):
            self.h[i + off] = z[i] * self.h[i + off] + r[1 ^ i]

        return np.concatenate(self.h)

    def backward(self, h_tm1, h_t):
        h_tm1, h_t = [self.vec_to_list(a) for a in [h_tm1, h_t]]

        res = self._solver(h_tm1, h_t)
        z = np.concatenate(res[:2])
        gru = np.concatenate(res[2:4])
        r = np.concatenate(res[4:6])

        # recover q
        q = gru / (1. - z)

        print '-' * 20
        mat = self._mat(h_tm1, h_t).astype(np.float32)
        print np.linalg.cond(mat), np.linalg.cond(self._A_inv)
        print z
        print r
        print q
        print '-' * 20

        # undo activations
        h_tm1_v = np.concatenate(h_tm1)
        upper = isigmoid(np.concatenate([z, r])) - np.dot(self.V, h_tm1_v)
        lower = itanh(q) - np.dot(self.G, r * np.dot(self.P, h_tm1_v))

        b = np.concatenate([upper, lower])

        # multiply by A^{-1} to recover x_t
        x_t = np.dot(self._A_inv, b)
        return x_t

    def get_state(self):
        return np.concatenate(self.h)

    @classmethod
    def _build_solver(cls):
        if hasattr(cls, '_solver_cache'):
            return cls._solver_cache

        N = 6
        matrix = [[0] * N for _ in range(N)]

        h_tm1 = []
        h_t = []
        for row in range(6):
            z_idx = row % 2
            kind = row / 2

            h_t.append(sympy.symbols('h_t^(%d)' % row, real=True))
            h_tm1.append(sympy.symbols('h_{t-1}^(%d)' % row, real=True))

            matrix[row][z_idx] = h_tm1[-1]

            if kind == 0:       # GRU
                off = 2
                matrix[row][off + z_idx] = 1
            elif kind == 1:     # GILR
                off = 4
                matrix[row][off + z_idx] = 1
            else:               # X-GILR
                assert kind == 2
                off = 4
                matrix[row][off + (z_idx ^ 1)] = 1

        A = sympy.Matrix(matrix)
        b = sympy.Matrix(h_t)
        x = list(sympy.symbols('z0, z1, gru0, gru1, r0, r1'))
        res = sympy.linsolve((A, b), x)
        assert len(res) == 1
        res = next(iter(res))

        _solver = sympy.lambdify(h_tm1 + h_t, res)
        _mat = sympy.lambdify(h_tm1 + h_t, A)

        def wrapper(h_tm1_d, h_t_d, func):
            inp = h_tm1_d + h_t_d
            return func(*inp)

        cls._solver_cache = staticmethod(partial(wrapper, func=_solver))
        cls._mat = staticmethod(partial(wrapper, func=_mat))
        return cls._solver_cache

    def vec_to_list(self, h):
        if isinstance(h, list):
            assert len(h) == 6
            assert all(arr.size == (self.N / 6) for arr in h)
            return h

        assert isinstance(h, np.ndarray)
        assert len(h.shape) == 1 and h.shape[0] == self.N
        return np.split(h, 6)



if __name__ == '__main__':
    #np.random.seed(2017)
    np.seterr(all='raise')

    N = 6
    n_layers = 3
    m = Stack(n_layers, N, Gru)

    steps = 1
    x = np.random.randn(steps, N)
    states = [m.get_state()]
    for i in range(steps):
        states.append(m.forward(x[i]))

    # [steps + 1, layers, units]
    states = np.array(states)

    print 'forwards'
    print 'inputs'
    print x
    """
    for i in range(n_layers):
        print 'layer %d' % i
        print states[1:, i, :]
    """

    print
    print 'backwards'

    # take needed data to reverse
    inits = states[0, :, :]
    top = states[1:, -1, :]

    """
    print 'init states'
    print inits
    print 'top activations'
    print top
    """

    # [layers + 1, steps, units]
    res = m.backward(inits, top)

    print 'recovered inputs'
    print res[0]
    """
    for i in range(n_layers):
        print 'layer %d' % i
        print res[i + 1]
    """
