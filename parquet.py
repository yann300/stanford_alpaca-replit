import pandas as pd
import numpy as np

pq = pd.read_parquet('train-00000-of-00004.parquet', engine='fastparquet')
print(pq['content'].iloc[:1])

file = open("dump.txt", "w")

file.write(pq['content'].iloc[1556])