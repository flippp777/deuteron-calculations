import numpy as np
import matplotlib.pyplot as plt

def x_squared(x):
    return x**2

dx = 0.1
x = np.arange(-2,2 + dx, dx)

x_sq = x_squared(x)

d_x_sq_dx = np.gradient(x_sq)/dx

plt.figure()
plt.plot(x, x_sq)
plt.plot(x, d_x_sq_dx)
plt.plot(x, np.gradient(d_x_sq_dx)/dx)

plt.figure()
plt.plot(x, x)
plt.plot(x, np.gradient(x)/dx)
plt.show()