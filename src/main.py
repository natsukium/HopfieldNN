from hopfield import HopfieldNeuralNetwork
from utils import generate_image, add_noise


def main():
    xs = generate_image(batchsize=3)
    x = add_noise(xs, 0.2)
    HNN = HopfieldNeuralNetwork()
    HNN.train(xs)

    ys = HNN.recall(x)
    return ys


if __name__ == "__main__":
    main()
