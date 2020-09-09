import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

class DataGeneratorLinear1D(object):
    def __init__(self, a=2.0, b=1.0):
        self.a = a
        self.b = b

    def create_dataset(self, xmin=0.0, xmax=10.0, noise=2.0, n=100):
        self.x = xmin + np.random.rand(n, 1)*(xmax - xmin)
        self.t = self.a*self.x + self.b + np.random.randn(n, 1)*noise
        self.modelx = np.arange(xmin, xmax, (xmax-xmin)/100.0)[:, None]
        self.modely = self.a*self.modelx + self.b

    def plot_dataset(self, include_generator=True, estimation=None):
        plt.figure(figsize=(6, 6))
        plt.plot(self.x, self.t, 'o', label='data points')
        if include_generator:
            plt.plot(self.modelx, self.modely, 'r-', label='true model')
        if estimation is not None:
            plt.plot(estimation[0], estimation[1], 'm-', label='estimation')
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.legend()
        plt.show()

class DataGeneratorLogistic1D(DataGeneratorLinear1D):
    def __init__(self, a=2.0, b=-10.0):
        DataGeneratorLinear1D.__init__(self, a, b)

    def create_dataset(self, xmin=0.0, xmax=10.0, n=1000):
        DataGeneratorLinear1D.create_dataset(self, xmin, xmax, 0.0, n)
        self.t = sigmoid(self.t) > np.random.rand(n, 1)
        self.modely = sigmoid(self.modely)

    def plot_dataset(self, include_generator=True, estimation=None):
        plt.figure(figsize=(6, 6))
        plt.plot(self.x, self.t, 'o', label='data points')
        if include_generator:
            plt.plot(self.modelx, self.modely, 'r-', label='true model')
        if estimation is not None:
            plt.plot(estimation[0], estimation[1], 'm-', label='estimation')
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.legend()
        plt.show()

class DataGeneratorLinear(object):
    def __init__(self, a=[2.0], b=1.0):
        self.dim = len(a)
        self.a = np.array(a)[:, None]
        self.b = np.array([[b]])

    def create_dataset(self, xmin=0.0, xmax=10.0, noise=2.0, n=100):
        self.x = xmin + np.random.rand(n, self.dim)*(xmax - xmin)
        self.t = np.dot(self.x, self.a) + self.b + np.random.randn(n, self.dim)*noise
        # Crear un grid, solo si dim == 2:
        #self.modelx = np.arange(xmin, xmax, (xmax-xmin)/100.0)[:, None]
        #self.modely = self.a*self.modelx + self.b

    """
    def plot_dataset(self, include_generator=True, estimation=None):
        plt.figure(figsize=(6, 6))
        plt.plot(self.x, self.t, 'o', label='data points')
        if include_generator:
            plt.plot(self.modelx, self.modely, 'r-', label='true model')
        if estimation is not None:
            plt.plot(estimation[0], estimation[1], 'm-', label='estimation')
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.legend()
        plt.show()
    """
