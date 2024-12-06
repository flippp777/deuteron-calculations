import matplotlib.pyplot as plt
import numpy as np
from asymmetric_uncertainties import aufloat

u = aufloat(3, 1.1, 1.0)
w = aufloat(3, 1.0, 1)

print(u)