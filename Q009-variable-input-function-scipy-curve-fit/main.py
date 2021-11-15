from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin

from pprint import pprint

# question: https://stackoverflow.com/questions/69814773/variable-input-function-scipy-curve-fit


class ShapeFunction(ABC):
    def __init__(self, params_count: int):
        self.params_count = params_count
        self.params = [0] * params_count

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, params: List[float]):
        if len(params) != self.params_count:
            raise ValueError(f"params count must be {self.params_count}")
        self.__params = params

    @abstractmethod
    def evaluate(self, t: List[float]) -> List[float]:
        pass


class Gaussian(ShapeFunction):
    def __init__(self):
        params_count = 3
        super().__init__(params_count)

    def evaluate(self, t: List[float]) -> List[float]:
        return self.Gaussian(t, *self.params)

    @staticmethod
    def Gaussian(x, mu, sigma, scale):
        return scale/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)


class ZeroOrder(ShapeFunction):
    def __init__(self):
        params_count = 1
        super().__init__(params_count)

    def evaluate(self, t: List[float]) -> List[float]:
        return self.ZeroOrder(t, *self.params)

    @staticmethod
    def ZeroOrder(x, a):
        return a


class Linear(ShapeFunction):
    def __init__(self):
        params_count = 2
        super().__init__(params_count)

    def evaluate(self, t: List[float]) -> List[float]:
        return self.Linear(t, *self.params)

    @staticmethod
    def Linear(x, a, b):
        return a+b*x


class Quadratic(ShapeFunction):
    def __init__(self):
        params_count = 3
        super().__init__(params_count)

    def evaluate(self, t: List[float]) -> List[float]:
        return self.Quadratic(t, *self.params)

    @staticmethod
    def Quadratic(x, a, b, c):
        return a+b*x+c*x**2


f1 = Gaussian()
f1.params = [0, 1, 10]

f2 = ZeroOrder()
f2.params = [1]

f3 = Linear()
f3.params = [1, 1]

f4 = Quadratic()
f4.params = [1, 1, 1]


shape_functions = [f1, f2, f3, f4]
coefs_count = sum([func.params_count for func in shape_functions])

original_coefs = []
for func in shape_functions:
    original_coefs.extend(func.params)

x = np.linspace(-5, 5, 100)
sample = f1.evaluate(x) + f2.evaluate(x) + f3.evaluate(x) + f4.evaluate(x)


def cost_function(coefs: list[float], *params):
    funcs: List[ShapeFunction] = params[0]
    parametric_values: List[float] = params[1]
    obj_y: List[float] = params[2]

    y = [0] * len(parametric_values)

    for func in funcs:
        func.params = coefs[:func.params_count]
        y += func.evaluate(parametric_values)

        coefs = coefs[func.params_count:]

    return np.sum((y - obj_y)**2)


x0 = [1] * coefs_count
solution = fmin(cost_function, x0, args=(shape_functions, x, sample))


for sol, coef in zip(solution, original_coefs):
    print(f"Original: {coef:.2f} -> solution: {sol:.2f}")


y_sol = np.sum([func.evaluate(x) for func in shape_functions], axis=0)

plt.plot(x, sample, label="Original")
plt.plot(x, y_sol, label="Solution", alpha=0.5, marker=".")

plt.legend()

plt.show()
