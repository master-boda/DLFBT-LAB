import numpy as np

def create_dataset_regression_1D(a=2.0, b=1.0, xmin=0.0, xmax=10.0, noise=2.0, n=100):
    x = xmin + np.random.rand(n, 1)*(xmax - xmin)
    t = a*x + b + np.random.randn(n, 1)*noise

    return x, t
