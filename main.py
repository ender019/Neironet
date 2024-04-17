import numpy as np
import matplotlib.pyplot as plt


class neironet():
    def __init__(self, inpp=3, outp=1, *args):
        np.random.seed(1)
        self.al = 0.009
        self.n = len(args)
        print(len(args))
        sizes = [inpp] + [e for e in args] + [outp]
        self.w=[2 * np.random.random((sizes[i], sizes[i+1])) - 1 for i in range(self.n + 1)]

    def relu(self, a): return a * (a > 0)
    def relu_d(self, a): return a > 0

    def sigmoid(self, a): return 1 / (1 + np.exp(-a))
    def sigmoid_d(self,a): return a * (1 - a)

    def tanh(self, a): return np.tanh(a)
    def tanh_d(self, a): return 1 - (a**2)

    def softmax(self, a):
        temp = np.exp(a)
        return temp / np.sum(temp, axis = 1, keepdims = True)
    def softmax_d(self, a):
        return a / len(a)

    def run(self, par):
        layers = [[par]]
        for i in range(self.n):
            layers.append(self.relu(layers[i] @ self.w[i]))
        layers.append(layers[-1] @ self.w[-1])
        return layers[-1]

    def train(self, inp, out):
        err_arr = []; cor_arr = []
        for j in range(100):
            err = 0; cor = 0
            for g in range(len(out)):
                dropmask = []
                layers = [inp[g: g + 1]]
                for i in range(self.n):
                    layers.append(self.relu(layers[i] @ self.w[i]))
                    dropmask.append(np.random.randint(2, size=layers[i+1].shape) * 2)
                layers.append(layers[-1] @ self.w[-1])

                err += np.sum((layers[-1] - out[g: g + 1]) ** 2)
                cor += int((layers[-1][0][0] >= 0.5) == out[g][0])

                layers_d = [layers[-1] - out[g: g + 1]]
                for i in range(self.n, 0, -1):
                    layers_d.append((layers_d[-1] @ self.w[i].T) * self.relu_d(layers[i]))
                    layers_d[-1]*=dropmask[i-1];
                layers_d.reverse()

                for i in range(self.n, -1, -1): self.w[i] -= self.al * (layers[i].T @ layers_d[i])
            err_arr.append(err)
            cor_arr.append(cor)
        print(err)
        plt.plot(err_arr)
        plt.plot(cor_arr)
        plt.show()


if __name__ == "__main__":
    net = neironet(3, 1, 6, 6, 6)
    inp = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1]])
    out = np.array([[1], [1], [0], [0]])
    net.train(inp, out)
    for i in range(len(out)): print(out[i], *net.run(inp[i]), '', sep='\n')
    print(*net.run([0, 1, 0]), 0)
    print(*net.run([1, 1, 1]), 1)
    print(*net.run([1, 1, 0]), 1)
