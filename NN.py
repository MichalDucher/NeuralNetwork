import numpy as np

class Network:

    def __init__(self, lr=0.1, n_iter=500, input_shape=784, output_shape=10):
        self.lr = lr
        self.n_inter = n_iter
        self.W1 = np.random.rand(10, input_shape) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.W2 = np.random.rand(output_shape, 10) - 0.5
        self.b2 = np.random.rand(output_shape, 1) - 0.5


    def fit(self, X, y):
        for i in range(self.n_inter):
            h1, L1, h2, L2 = self._forward_prop(X)
            dW1, db1, dW2, db2 = self._backward_prop(h1, L1, h2, L2, X, y)
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            if(i % 10 == 0):
                predictions = self._get_predictions(L2)
                print('accuracy: ', self.get_accuracy(predictions, y))


    def _forward_prop(self, X):
        h1 = self.W1.dot(X) + self.b1
        L1 = self._ReLU(h1)
        h2 = self.W2.dot(L1) + self.b2
        L2 = self._soft_max(h2)
        return h1, L1, h2, L2

    def _backward_prop(self, h1, L1, h2, L2, X, y):
        m = y.size
        one_hot_y = self._one_hot(y)
        dh2 = L2 - one_hot_y
        dW2 = 1 / m * dh2.dot(L1.T)
        db2 = 1 / m * np.sum(dh2)
        dh1 = self.W2.T.dot(dh2) * self._deriv_ReLU(h1)
        dW1 = 1 / m * dh1.dot(X.T)
        db1 = 1 / m * np.sum(dh1)
        return dW1, db1, dW2, db2

    def _get_predictions(self, L2):
        return np.argmax(L2, 0)

    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def predict(self, X):
        _, _, _, L2 = self._forward_prop(X)
        predictions = self._get_predictions(L2)
        return predictions

    def _one_hot(self, y):
        one_hot = np.zeros((y.size, y.max() + 1))
        one_hot[np.arange(y.size), y] = 1
        return one_hot.T

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def _soft_max(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def _ReLU(self, x):
        return np.maximum(x, 0)

    def _deriv_ReLU(self, x):
        return x > 0

