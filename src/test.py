import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
def f(x):
    return x**2

plt.plot(x, f(x))
plt.plot(x, np.gradient(f(x), x))
plt.plot(x, np.gradient(np.gradient(f(x), x), x))
plt.show()