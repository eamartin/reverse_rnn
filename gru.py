from collections import OrderedDict
import copy
from pprint import pprint
import numpy as np
import sympy

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def isigmoid(x):
    return -np.log((1. / x) - 1.)

def itanh(x):
    return 0.5 * np.log((1. + x) / (1. - x))

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

        # undo activations
        h_tm1_v = np.concatenate(h_tm1)
        upper = isigmoid(np.concatenate([z, r])) - np.dot(self.V, h_tm1_v)
        lower = itanh(q) - np.dot(self.G, r * np.dot(self.P, h_tm1_v))

        b = np.concatenate([upper, lower])
        A = np.vstack([self.U, self.W])

        # solve NxN linear system. Note this can be done more efficiently by
        # inverting A once at the beginning of backwards pass
        x_t = np.linalg.solve(A, b)
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

            h_t.append(sympy.symbols('h_t^(%d)' % row))
            h_tm1.append(sympy.symbols('h_{t-1}^(%d)' % row))

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
        def solver(h_tm1_d, h_t_d):
            inp = h_tm1_d + h_t_d
            return _solver(*inp)

        cls._solver_cache = staticmethod(solver)
        return solver

    def vec_to_list(self, h):
        if isinstance(h, list):
            assert len(h) == 6
            assert all(arr.size == (self.N / 6) for arr in h)
            return h

        assert isinstance(h, np.ndarray)
        assert len(h.shape) == 1 and h.shape[0] == self.N
        return np.split(h, 6)


class GruStack(object):
    def __init__(self, n_layers, hidden_size):
        self.N = hidden_size
        self.grus = [Gru(hidden_size) for i in range(n_layers)]

    def forward(self, x):
        assert x.size == self.N
        net = x
        for gru in self.grus:
            net = gru.forward(net)
        return self.get_state()

    def backward(self, inits, top_act):
        assert inits.shape[0] == len(self.grus)
        assert inits.shape[1] == top_act.shape[1] == self.N

        upper_act = top_act
        all_acts = [np.stack(upper_act, axis=0)]
        for layer_idx, gru in reversed(list(enumerate(self.grus))):
            lower_act = []
            for i in range(steps):
                back = upper_act[i - 1] if i != 0 else inits[layer_idx]
                now = upper_act[i]
                lower_act.append(gru.backward(back, now))

            all_acts.append(np.stack(lower_act, axis=0))
            upper_act = lower_act

        return np.array(list(reversed(all_acts)))

    def get_state(self):
        return np.stack([g.get_state() for g in self.grus], axis=0)

if __name__ == '__main__':
    np.random.seed(2017)
    N = 6
    n_layers = 4
    m = GruStack(n_layers, N)

    steps = 2
    x = np.random.randn(steps, N)
    states = [m.get_state()]
    for i in range(steps):
        states.append(m.forward(x[i]))

    # [steps + 1, layers, units]
    states = np.array(states)

    print 'forwards'
    print 'inputs'
    print x
    for i in range(n_layers):
        print 'layer %d' % i
        print states[1:, i, :]

    print
    print 'backwards'

    # take needed data to reverse
    inits = states[0, :, :]
    top = states[1:, -1, :]

    print 'init states'
    print inits
    print 'top activations'
    print top

    # [layers + 1, steps, units]
    res = m.backward(inits, top)

    print 'recovered inputs'
    print res[0]
    for i in range(n_layers):
        print 'layer %d' % i
        print res[i + 1]
