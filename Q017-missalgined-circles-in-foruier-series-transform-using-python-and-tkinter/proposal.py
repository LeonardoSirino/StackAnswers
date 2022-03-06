from numbers import Complex
from typing import Callable, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def fourier_series_coeff_numpy(f: Callable, T: float, N: int) -> List[Complex]:
    """Get the coefficients of the Fourier series of a function.

    Args:
        f (Callable): function to get the Fourier series coefficients of.
        T (float): period of the function.
        N (int): number of coefficients to get.

    Returns:
        List[Complex]: list of coefficients of the Fourier series.
    """
    f_sample = 2 * N

    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.fft(f(t)) / t.size

    return y


def evaluate_fourier_series(coeffs: List[Complex], ang: float, period: float) -> List[Complex]:
    """Evaluate a Fourier series at a given angle.

    Args:
        coeffs (List[Complex]): list of coefficients of the Fourier series.
        ang (float): angle to evaluate the Fourier series at.
        period (float): period of the Fourier series.

    Returns:
        List[Complex]: list of complex numbers representing the Fourier series.
    """
    N = np.fft.fftfreq(len(coeffs), d=1/len(coeffs))
    N = filter(lambda x: x >= 0, N)

    y = 0
    radius = []
    for n, c in zip(N, coeffs):
        r = 2 * c * np.exp(1j * n * ang / period)
        y += r

        radius.append(r)

    return radius


def square_function_factory(period: float):
    """Builds a square function with given period.

    Args:
        period (float): period of the square function.
    """
    def f(t):
        if isinstance(t, Iterable):
            return [1.0 if x % period < period / 2 else -1.0 for x in t]
        elif isinstance(t, float):
            return 1.0 if t % period < period / 2 else -1.0

    return f


def saw_tooth_function_factory(period: float):
    """Builds a saw-tooth function with given period.
    
    Args:
        period (float): period of the saw-tooth function.
    """
    def f(t):
        if isinstance(t, Iterable):
            return [1.0 - 2 * (x % period / period) for x in t]
        elif isinstance(t, float):
            return 1.0 - 2 * (t % period / period)

    return f


def main():
    PERIOD = 1
    GRAPH_RANGE = 3.0
    N_COEFFS = 30

    f = square_function_factory(PERIOD)
    # f = lambda t: np.sin(2 * np.pi * t / PERIOD)
    # f = saw_tooth_function_factory(PERIOD)

    coeffs = fourier_series_coeff_numpy(f, 1, N_COEFFS)
    radius = evaluate_fourier_series(coeffs, 0, 1)

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 5))

    ang_cum = []
    amp_cum = []

    for ang in np.linspace(0, 2*np.pi * PERIOD * 3, 200):
        radius = evaluate_fourier_series(coeffs, ang, 1)

        x = np.cumsum([x.imag for x in radius])
        y = np.cumsum([x.real for x in radius])

        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)

        axs[0].plot(x, y)
        axs[0].set_ylim(-GRAPH_RANGE, GRAPH_RANGE)
        axs[0].set_xlim(-GRAPH_RANGE, GRAPH_RANGE)

        ang_cum.append(ang)
        amp_cum.append(y[-1])

        axs[1].plot(ang_cum, amp_cum)

        axs[0].axhline(y=y[-1],
                       xmin=x[-1] / (2 * GRAPH_RANGE) + 0.5,
                       xmax=1.2,
                       c="black",
                       linewidth=1,
                       zorder=0,
                       clip_on=False)

        min_x, max_x = axs[1].get_xlim()
        line_end_x = (ang - min_x) / (max_x - min_x)

        axs[1].axhline(y=y[-1],
                       xmin=-0.2,
                       xmax=line_end_x,
                       c="black",
                       linewidth=1,
                       zorder=0,
                       clip_on=False)

        plt.pause(0.01)

        axs[0].clear()
        axs[1].clear()


if __name__ == '__main__':
    main()
