import numpy as np


def generate_image(n=5, batchsize=1):
    return np.random.choice([-1, 1], size=(batchsize, n**2))


def add_noise(x, p):
    p = [p, 1-p]
    return x * np.random.choice([-1, 1], size=x.shape, p=p)
