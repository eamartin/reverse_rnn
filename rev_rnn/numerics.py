import numpy as np

def _t_to_s(tanh):
    typ = tanh.dtype.type
    return (tanh + typ(1)) / typ(2)

def _s_to_t(sig):
    typ = sig.dtype.type
    return typ(2) * sig - typ(1)

def sigmoid(x):
    return _t_to_s(np.tanh(x))

def itanh(tanh):
    eps = np.finfo(tanh.dtype).epsneg
    one = tanh.dtype.type(1)
    tanh = np.clip(tanh, -one + eps, one - eps)
    return np.arctanh(tanh)

def isigmoid(p):
    return itanh(_s_to_t(p))
