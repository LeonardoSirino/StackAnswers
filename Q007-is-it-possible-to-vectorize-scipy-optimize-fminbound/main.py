import multiprocessing
from multiprocessing import Pool, Process
from pprint import pprint
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import fminbound

data = np.array(
    [
        (212.82275865, 1650.40828168, 0., 0),
        (214.22056952, 1649.99898924, 10.38, 0),
        (212.86786868, 1644.25228805, 116.288, 0),
        (212.78680031, 1643.87461108, 122.884, 0),
        (212.57489485, 1642.01124032, 156.313, 0),
        (212.53483954, 1641.61858242, 162.618, 0),
        (212.43922274, 1639.58782771, 196.314, 0),
        (212.53726315, 1639.13842423, 202.619, 0),
        (212.2888428, 1637.24641296, 236.306, 0),
        (212.2722447, 1636.92307229, 242.606, 0),
        (212.15559302, 1635.0529813, 276.309, 0),
        (212.17535631, 1634.60618711, 282.651, 0),
        (212.02545613, 1632.72139574, 316.335, 0),
        (211.99988779, 1632.32053329, 322.634, 0),
        (211.33419846, 1631.07592039, 356.334, 0),
        (211.58972239, 1630.21971902, 362.633, 0),
        (211.70332122, 1628.2088542, 396.316, 0),
        (211.74610735, 1627.67591368, 402.617, 0),
        (211.88819518, 1625.67310022, 436.367, 0),
        (211.90709414, 1625.39410321, 442.673, 0),
        (212.00090348, 1623.42655008, 476.332, 0),
        (211.9249017, 1622.94540583, 482.63, 0),
        (212.34321938, 1616.32949197, 597.329, 0),
        (213.09638942, 1615.2869643, 610.4, 0),
        (219.41313491, 1580.22234313, 1197.332, 0),
        (220.38660128, 1579.20043302, 1210.37, 0),
        (236.35472669, 1542.30863041, 1798.267, 0),
        (237.41755384, 1541.41679119, 1810.383, 0),
        (264.08373622, 1502.66620597, 2398.244, 0),
        (265.65655239, 1501.64308908, 2410.443, 0),
        (304.66999824, 1460.94068336, 2997.263, 0),
        (306.22550945, 1459.75817211, 3010.38, 0),
        (358.88879764, 1416.472238, 3598.213, 0),
        (361.14046402, 1415.40942931, 3610.525, 0),
        (429.96379858, 1369.7972467, 4198.282, 0),
        (432.06565776, 1368.22265539, 4210.505, 0),
        (519.30493383, 1319.01141844, 4798.277, 0),
        (522.12134083, 1317.68234967, 4810.4, 0),
        (630.00294242, 1265.05368942, 5398.236, 0),
        (633.67624272, 1263.63633508, 5410.431, 0),
        (766.29767476, 1206.91262814, 5997.266, 0),
        (770.78300649, 1205.48393374, 6010.489, 0),
        (932.92308019, 1145.87780431, 6598.279, 0),
        (937.54373403, 1141.55438694, 6609.525, 0),
    ], dtype=[
        ('x', 'f8'), ('y', 'f8'), ('t', 'f8'), ('dmin', 'f8'),
    ]
)

coeffs = np.polyfit(
    data['t'], pd.DataFrame(data[['x', 'y']]).values, 3
)


def curve(t):
    x = np.polyval(coeffs[:, 0], t)
    y = np.polyval(coeffs[:, 1], t)

    return x, y


def f(t, p):
    x, y = curve(t)
    return np.hypot(x - p['x'], y - p['y'])


def get_min_distance(point: np.array) -> np.array:
    tmin = fminbound(f, -50, 6659.525, args=(point, ))
    dmin = f(tmin, point)

    return dmin


def get_distances(data: np.array, index: int, response_dict: Dict) -> List[float]:
    d_mins = []
    for point in data:
        dmin = get_min_distance(point)
        d_mins.append(dmin)

    response_dict[index] = d_mins


def parallel_code() -> List[float]:
    processes = 4
    chunck_size = int(len(data) / processes)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    jobs = []

    for i in range(processes):
        p = Process(target=get_distances,
                    args=(data[i * chunck_size: (i + 1) * chunck_size],
                          i,
                          return_dict))
        p.start()
        jobs.append(p)

    for p in jobs:
        p.join()

    dmins = return_dict.values()
    dmins = np.concatenate(dmins)

    return dmins


def original_code() -> List[float]:
    d_mins = []
    for point in data:
        dmin = get_min_distance(point)
        d_mins.append(dmin)

    return d_mins


if __name__ == '__main__':
    d1 = parallel_code()
    d2 = original_code()

    print(np.array(d1) - np.array(d2))

    import timeit

    original_time = timeit.timeit('original_code()',
                                  globals=globals(),
                                  number=1)
    print(f'Original time: {original_time:.3f}')

    parallel_time = timeit.timeit('parallel_code()',
                                  globals=globals(),
                                  number=1)
    print(f'Parallel time: {parallel_time:.3f}')
