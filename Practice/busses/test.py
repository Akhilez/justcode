import pandas as pd
import pandas_profiling
import numpy as np

b43 = pd.read_excel('data/R43RidersByTripJ7to9.xlsx', sheet_name='R43RidersByTripJ7to9')

pandas_profiling.ProfileReport(b43)



