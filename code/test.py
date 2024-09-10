import numpy as np
from scipy.special import genlaguerre

a = 0

ar = np.array([1,1,1,1])

for i in range(4):
    a += ar
    print(a)
