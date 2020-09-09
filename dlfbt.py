import numpy as np
import matplotlib.pyplot as plt

class DataGeneratorRegression1D(object):
    def __init__(self, a=2.0, b=1.0):
        self.a = a
        self.b = b

    def create_dataset(self, xmin=0.0, xmax=10.0, noise=2.0, n=100):
        self.x = xmin + np.random.rand(n, 1)*(xmax - xmin)
        self.t = self.a*self.x + self.b + np.random.randn(n, 1)*noise
        self.modelx = np.arange(xmin, xmax, (xmax-xmin)/100.0)
        self.modely = self.a*self.modelx + self.b

    def plot_dataset(self, include_model=True):
        plt.figure(figsize=(6, 6))
        plt.plot(self.x, self.t, 'o')
        if include_model:
            plt.plot(self.modelx, self.modely, 'r-')
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.show()

    
