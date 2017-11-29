import numpy as np

class Stack(object):
    def __init__(self, n_layers, hidden_size, cls):
        self.N = hidden_size
        self.layers = [cls(hidden_size) for i in range(n_layers)]

    def forward(self, x):
        assert x.size == self.N
        net = x
        for layer in self.layers:
            net = layer.forward(net)
        return self.get_state()

    def backward(self, inits, top_act):
        assert inits.shape[0] == len(self.layers)
        assert inits.shape[1] == top_act.shape[1] == self.N
        steps = top_act.shape[0]

        upper_act = top_act
        all_acts = [np.stack(upper_act, axis=0)]
        for layer_idx, layer in reversed(list(enumerate(self.layers))):
            lower_act = []
            for i in range(steps):
                back = upper_act[i - 1] if i != 0 else inits[layer_idx]
                now = upper_act[i]
                lower_act.append(layer.backward(back, now))

            all_acts.append(np.stack(lower_act, axis=0))
            upper_act = lower_act

        return np.array(list(reversed(all_acts)))

    def get_state(self):
        return np.stack([g.get_state() for g in self.layers], axis=0)
