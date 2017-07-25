

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib as plt

#Import Dataset
df = pd.read_csv('../data/train.csv')
df1 = pd.read_csv('../data/test.csv')


#Select numerical data types
set(df.dtypes.tolist())
dfnum = df.select_dtypes(include = ['float64', 'int64'])
dfnum.info()