import numpy as np


class HopfieldNeuralNetwork(object):
    def train(self, xs):
        self.w = np.einsum('ij,ik->jk', xs, xs)
        self.w[np.eye(xs.shape[1]) == 1] = 0
        return self.w / xs.shape[0]

    def recall(self, x, threshold=None):
        if threshold is None:
            self.threshold = np.zeros(x.shape)
        else:
            self.threshold = threshold
        delta = 1e-8

        E = self._energy(x)
        while True:
            x = np.sign(self.w@x - self.threshold)
            E_new = self._energy(x)
            if abs(E_new - E) < delta:
                break
            E = E_new

        return x

    def _energy(self, x):
        return - np.einsum('ij,i,j', self.w, x, x) / 2 \
            + np.sum(self.threshold * x)
