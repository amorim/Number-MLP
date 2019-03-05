import numpy as np
import random


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def get_zeros(arr):
    ret = []
    for i in arr:
        ret.append(np.zeros(i.shape))
    return ret


class Network:

    def __init__(self, sizes, ansmap):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []
        self.best = float('-inf')
        for i in sizes[1:]:
            self.biases.append(np.random.randn(i, 1))
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i + 1], sizes[i]))
        self.ansmap = ansmap

    def output(self, activations):
        for b, w in zip(self.biases, self.weights):
            activations = sigmoid(np.dot(w, activations) + b)
        return activations

    def grad_descent(self, training, loops, bsize, rate, test_data=None):
        n = len(training)
        for j in range(loops):
            print('Now training the network. Pass ' + str(j + 1) + ' out of ' + str(loops) + '.')
            random.shuffle(training)
            batches = []
            for i in range(0, n, bsize):
                batches.append(training[i:i + bsize])
            for batch in batches:
                self.update_batch(batch, rate)
            print('Pass ' + str(j + 1) + ' out of ' + str(loops) + ' completed.')
            if test_data:
                evaluation = self.evaluate(test_data)
                if evaluation > self.best:
                    print('Found better: ' + str(evaluation))
                    self.best = evaluation
                    self.export()



    def update_batch(self, batch, rate):
        grad_b, grad_w = self.get_grads_array()
        for x, y in batch:
            diff_grad_b, diff_grad_w = self.backpropagation(x, y)
            for i in range(len(grad_b)):
                grad_b[i] += diff_grad_b[i]
            for i in range(len(grad_w)):
                grad_w[i] += diff_grad_w[i]
        for i in range(len(grad_w)):
            self.weights[i] -= (rate / len(batch)) * grad_w[i]
        for i in range(len(grad_b)):
            self.biases[i] -= (rate / len(batch)) * grad_b[i]

    def backpropagation(self, x, y):
        grad_b, grad_w = self.get_grads_array()
        activation = x
        activations = [x]
        zresults = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zresults.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.dcost(activations[-1], y) * dsigmoid(zresults[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zresults[-l]
            sp = dsigmoid(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return grad_b, grad_w

    def evaluate(self, data):
        res = []
        for x, y in data:
            out = np.argmax(self.output(x))
            res.append((out, self.ansmap[y]))
            #print(res)
        return sum(int(x == y) for (x, y) in res)

    def dcost(self, activations, y):
        return activations - y

    def export(self):
        np.save('biases', self.biases)
        np.save('weights', self.weights)

    def load(self, biases_name, weights_name):
        self.biases = np.load(biases_name)
        self.weights = np.load(weights_name)

    def get_grads_array(self):
        return get_zeros(self.biases), get_zeros(self.weights)


