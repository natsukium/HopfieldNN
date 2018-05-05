import numpy as np


class HopfieldNeuralNetwork(object):
    def train(self, xs):
        self.w = np.einsum('ij,ik->jk', xs, xs)
        self.w[np.eye(xs.shape[1]) == 1] = 0
        return self.w / xs.shape[0]

    def recall(self, xs, threshold=None):
        if threshold is None:
            self.threshold = np.zeros(xs.shape[1])
        else:
            self.threshold = threshold
        delta = 1e-8

        ys = []
        for x in xs:
            E = self._energy(x)
            while True:
                x = np.sign(self.w@x - self.threshold)
                E_new = self._energy(x)
                if abs(E_new - E) < delta:
                    break
                E = E_new
            ys.append(x)

        return ys

    def _energy(self, x):
        return - np.einsum('ij,i,j', self.w, x, x) / 2 \
            + np.sum(self.threshold * x)
