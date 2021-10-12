import numpy as np
import pandas as pd

values = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
          np.nan, np.nan, np.nan, np.nan, np.nan, 'WH042700', 90510000, 90510000]

df = pd.DataFrame(values)

first_non_na = df.dropna().iloc[0,0]

print(first_non_na)
