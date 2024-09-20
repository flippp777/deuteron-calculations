import numpy as np
import matplotlib.pyplot as plt

def x_squared(x):
    return x**2

dx = 0.1
x = np.arange(-2,2 + dx, dx)
print(x)

def derivative(x, f, n):
    dx = abs(x[0] - x[1])
    
    for i in range(n):
        f = np.gradient(f) / dx
        
    return f

print(derivative(x, x, 2))

