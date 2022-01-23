from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd

data = """ID  date     growth  t
x1a 1/1/2018    1.2  0
x1a 2/1/2018    1    1
x1a 3/1/2018    3    2
x1a 4/1/2018    2    3
x1a 5/1/2018    0.9  4
z8d 3/1/2018    0.7  0
z8d 3/2/2018    1    1
z8d 3/3/2018    0.8  2
z8d 3/4/2018    0.6  3
z8d 3/5/2018    2.3  4
z8d 3/6/2018    1.7  5
z8d 3/7/2018    1    6
z8d 3/8/2018    2.1  7
j2u 1/1/2020    0.9  0
j2u 1/2/2020    0.8  1
j2u 1/3/2020    1.3  2
j2u 1/4/2020    1.4  3
j2u 1/5/2020    2    4
j2u 1/6/2020    1.4  5"""

df = pd.read_csv(StringIO(data), sep='\s+')
df['date'] = pd.to_datetime(df['date'])

for id_, df in df.groupby(by='ID'):
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    plt.plot(df.index + 1, df['growth'], label=id_)


plt.legend()
plt.xlabel('Index')
plt.ylabel('Growth')
plt.show()
