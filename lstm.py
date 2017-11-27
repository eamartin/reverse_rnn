from collections import OrderedDict
import copy
from pprint import pprint
import numpy as np
import sympy

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

class Lstm(object):
    def __init__(self, hidden_size):
        self.N = N = hidden_size
        assert N % 2 == 0
        self.W = np.random.randn(4 * N, 2 * N)

        self.h = np.zeros(N)

        # 3X f0, 3X f1. 1X i0z0, 1X i1z1, 2X i0z1, 2X i1z0
        self.c = OrderedDict(
            (p, np.random.randn(N / 2)) for p in [
                (0, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 1),
                (1, 0, 1),
                (1, 1, 0)
            ]
        )

        self._lr_solver = build_solver(self.c.keys())
        self._iz_solver = self._gen_iz_solver()

    def combine_c(self, upper=True):
        i = 0 if upper else 1
        c = [v for (idx, v) in self.c.items() if idx[0] == i]
        assert len(c) == 3
        return np.mean(np.stack(c, axis=1), axis=1)

    def forward(self, x):
        assert x.size == self.N
        inp = np.concatenate([self.h, x], axis=0)
        preact = np.dot(self.W, inp)

        f, i, z, o = np.split(preact, 4)
        f = np.split(sigmoid(f), 2)
        i = np.split(sigmoid(i), 2)
        z = np.split(np.tanh(z), 2)
        o = sigmoid(o)

        print f
        print i
        print z

        for idx in self.c.keys():
            this_z = z[idx[2]]
            if idx[1] != idx[2]:
                pass
                #this_z *= (np.abs(this_z) ** .1)
            self.c[idx] = f[idx[0]] * self.c[idx] + i[idx[1]] * this_z

        c = np.concatenate([
            self.combine_c(upper=True),
            self.combine_c(upper=False)
        ])
        self.h = o * c

    @staticmethod
    def _gen_iz_solver():
        A = sympy.Matrix(
            [[1, 0, 1, 0],
             [1, 0, 0, 1.1],
             [0, 1, 1.1, 0],
             [0, 1, 0, 1]]
        )

        x = list(sympy.symbols('li0, li1, lz0, lz1'))
        b = sympy.Matrix(sympy.symbols('li0z0, li0z1, li1z0, li1z1'))
        res = sympy.linsolve((A, b), x)
        assert len(res) == 1
        res = next(iter(res))
        return sympy.lambdify(list(b), res)

    def backward(self, c_tm1, c_t):
        # both inputs are OrderedDicts from parity to vec
        res = self._lr_solver(c_tm1.values(), c_t.values())
        f0, f1 = res[:2]

        print 'backwards'
        print f0, f1

        # now need to untangle iz terms. can do this because we know i>=0.
        # This allows taking logs and solving simple 4x4 linear system

        iz = np.array(res[2:])
        sign = np.sign(iz)
        liz = np.log(np.abs(iz))
        iz_res = self._iz_solver(*liz)
        print iz_res

def build_solver(parity):
    N = len(parity)
    assert N == 6
    matrix = [[0] * N for _ in range(N)]

    h_tm1 = []
    h_t = []
    for row, idx in enumerate(parity):
        f_idx = idx[0]
        iz_idx = 2 * idx[1] + idx[2]

        h_t.append(sympy.symbols('h_t^(%d)' % row))
        h_tm1.append(sympy.symbols('h_{t-1}^(%d)' % row))

        matrix[row][f_idx] = h_tm1[-1]
        matrix[row][2 + iz_idx] = 1

    A = sympy.Matrix(matrix)
    b = sympy.Matrix(h_t)
    x = list(sympy.symbols('f0, f1, i0z0, i0z1, i1z0, i1z1'))
    res = sympy.linsolve((A, b), x)
    assert len(res) == 1
    res = next(iter(res))

    #sympy.pprint(A)

    _solver = sympy.lambdify(h_tm1 + h_t, res)
    def solver(h_tm1_d, h_t_d):
        #d = {h_tm1[i]: h_tm1_d[i][0] for i in range(N)}
        #sympy.pprint(A.subs(d))
        inp = h_tm1_d + h_t_d
        return _solver(*inp)

    return solver

if __name__ == '__main__':
    N = 4
    m = Lstm(N)
    c_tm1 = copy.copy(m.c)

    x = np.random.randn(N)
    m.forward(x)

    m.backward(c_tm1, m.c)
