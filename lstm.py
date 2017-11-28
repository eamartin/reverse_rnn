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

class Lstm(object):
    def __init__(self, hidden_size):
        self.N = N = hidden_size
        assert N % 2 == 0
        self.W = np.random.randn(4 * N, 2 * N)

        self.h = np.random.randn(N)

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

    @classmethod
    def combine_c_half(cls, c_dict, upper=True):
        i = 0 if upper else 1
        c = [v for (idx, v) in c_dict.items() if idx[0] == i]
        assert len(c) == 3
        return np.mean(np.stack(c, axis=1), axis=1)

    @classmethod
    def combine_c(cls, c_dict):
        return np.concatenate([
            cls.combine_c_half(c_dict, upper=True),
            cls.combine_c_half(c_dict, upper=False)
        ])

    def forward(self, x):
        print 'forward'
        assert x.size == self.N
        inp = np.concatenate([self.h, x], axis=0)
        print 'inp:', inp
        preact = np.dot(self.W, inp)

        f, i, z, o = np.split(preact, 4)
        f = np.split(sigmoid(f), 2)
        i = np.split(sigmoid(i), 2)
        z = np.split(np.tanh(z), 2)
        o = sigmoid(o)

        print 'f:', f
        print 'i:', i
        print 'z:', z

        for idx in self.c.keys():
            this_z = z[idx[2]]
            if idx[1] != idx[2]:
                # hack to produce non-singular system for iz
                this_z = this_z * (np.abs(this_z) ** .1)

            self.c[idx] = f[idx[0]] * self.c[idx] + i[idx[1]] * this_z

        c = self.combine_c(self.c)
        self.h = o * c
        print 'h', self.h

    @classmethod
    def _gen_iz_solver(cls):
        # contains 1.1 values to be non-singular
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
        print 'backward'
        # both inputs are OrderedDicts from parity to vec
        res = self._lr_solver(c_tm1.values(), c_t.values())
        f = list(res[:2])

        print 'f:', f

        # now need to untangle iz terms. can do this because we know i>=0.
        # This allows taking logs and solving 4x4 linear system

        # iz = [i0z0, i0(z1^1.1), i1(z0^1.1), i1z1], (4, N/2) shape
        iz = np.array(res[2:])
        sign = np.sign(iz)
        assert np.all(sign[:2] == sign[2:])
        sign = sign[:2]

        liz = np.log(np.abs(iz))
        iz_res = np.exp(self._iz_solver(*liz))
        i, z = list(iz_res[:2]), list(iz_res[2:])
        z[0] *= sign[0]
        z[1] *= sign[1]
        # now have f, i, z recovered!
        print 'i:', i
        print 'z:', z

        # undo activations
        pre_f = isigmoid(np.concatenate(f))
        pre_i = isigmoid(np.concatenate(i))
        pre_z = itanh(np.concatenate(z))

        out = np.concatenate([pre_f, pre_i, pre_z])
        inp = np.linalg.lstsq(self.W[:3 * self.N], out)[0]
        print 'inp:', inp

        o = sigmoid(np.dot(self.W[-self.N:], inp))
        c = self.combine_c(c_t)
        h = o * c
        print 'h', h

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
    np.random.seed(2017)
    N = 4
    m = Lstm(N)
    c_tm1 = copy.copy(m.c)

    x = np.random.randn(N)
    m.forward(x)
    print
    m.backward(c_tm1, m.c)
