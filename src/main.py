from hopfield import HopfieldNeuralNetwork
from utils import generate_image, add_noise


def main():
    xs = generate_image(batchsize=3)
    x = add_noise(xs, 0.2)
    HNN = HopfieldNeuralNetwork()
    HNN.train(xs)

    n = xs.shape[0]
    return [HNN.recall(x[i]) for i in range(n)]


if __name__ == "__main__":
    main()
