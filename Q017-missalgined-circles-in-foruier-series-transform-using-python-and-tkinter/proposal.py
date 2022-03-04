from numbers import Complex
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def fourier_series_coeff_numpy(f, T, N):
    f_sample = 2 * N

    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.fft(f(t)) / t.size

    return y


def evaluate_fourier_series(coeffs: List[Complex], ang: float, period: float) -> Complex:
    N = np.fft.fftfreq(len(coeffs), d=1/len(coeffs))

    y = 0
    radius = []
    for n, c in zip(N, coeffs):
        r = c * np.exp(1j * n * ang / period)
        y += r

        radius.append(r)

    radius = filter(lambda x: abs(x) > 0.1, radius)
    radius = sorted(radius, key=lambda x: abs(x), reverse=True)

    return radius


coeffs = fourier_series_coeff_numpy(lambda t: np.sin(2 * np.pi * t), 1, 3)

for ang in np.linspace(0, 2*np.pi, 200):
    radius = evaluate_fourier_series(coeffs, ang, 1)

    x = np.cumsum([x.real for x in radius])
    y = np.cumsum([x.imag for x in radius])

    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)

    plt.plot(y, x)
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.pause(0.01)

    plt.clf()
