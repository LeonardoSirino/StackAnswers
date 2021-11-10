import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt

# question: https://stackoverflow.com/questions/69814773/variable-input-function-scipy-curve-fit


def Gaussian(x, mu, sigma, scale):
    return scale/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)


def ZeroOrder(x, a):
    return a


def Linear(x, a, b):
    return a+b*x


def SecondOrder(x, a, b, c):
    return a+b*x+c*x**2


x = np.linspace(-5, 5, 100)
sample = Gaussian(x, 0, 1, 10) + ZeroOrder(x, 1) + \
    Linear(x, 1, 1) + SecondOrder(x, 1, 1, 1)


plt.plot(x, sample)
plt.show()

# def Combined(x, *params):
#     off = Linear(x, params[0], params[1])
#     peak1 = Gaussian(x, params[2], params[3], params[4])
#     peak2 = Gaussian(x, params[5], params[6], params[7])
#     peak3 = Gaussian(x, params[8], params[9], params[10])
#     return off + peak1 + peak2 + peak3


# popt, pcov = opt.curve_fit(Combined, data[10][0], data[10][1], method='lm', check_finite=True, p0=[
#                            0.1, 0.1, 115, 508.33, 7.1, 130, 508.33, 7.1, 165.84, 508.33, 7.1])
