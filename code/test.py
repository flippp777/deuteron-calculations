import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,100,1000)
def exp(x, a, b):
    return a * np.exp(b * x)

plt.plot(x, exp(x, 1, -1))
plt.semilogy()
plt.show()