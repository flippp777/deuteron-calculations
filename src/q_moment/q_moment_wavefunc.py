import scipy as sp
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')
from make.position_space_wavefunc import deuteron_wavefunc

def derivative(x, f, n):
    dx = abs(x[0] - x[1])
    
    for i in range(n):
        f = np.gradient(f) / dx
        
    return f

def u(r, Nmax, aHO):
    return r * deuteron_wavefunc(Nmax, aHO, r)[1]

def w(r, Nmax, aHO):
    return r * deuteron_wavefunc(Nmax, aHO, r)[2]

def Q_quad(Nmax, aHO):
    return sp.integrate.quad(lambda r: 1/np.sqrt(50) * r**2 * u(r, Nmax, aHO) * w(r, Nmax, aHO) - 1/20 * r**2 * w(r, Nmax, aHO)**2, 0, 100)[0]

def q(r, Nmax, aHO):
    return 1/np.sqrt(50) * r**2 * u(r, Nmax, aHO) * w(r, Nmax, aHO) - 1/20 * r**2 * w(r, Nmax, aHO)**2

def Q_trapz(r, Nmax, aHO):
    return sp.integrate.trapezoid(q(r, Nmax, aHO), x=r)

r = np.arange(0,50,0.1)

plt.figure()
for Nmax in range(0,60,5):
    plt.title("u*w")
    plt.plot(u(r, Nmax, 1.4) * w(r, Nmax, 1.4) * r**2)
    plt.semilogy()
    plt.ylim(top=1,bottom=10**(-6))
    
plt.figure()
for Nmax in range(0,60,5):
    plt.title("w")
    plt.plot(w(r, Nmax, 1.4)**2 * r**2)
    plt.semilogy()
    plt.ylim(top=1,bottom=10**(-6))
    
plt.figure()
for Nmax in range(0,60,2):
    print(Q_trapz(r, Nmax, 1.4))
    plt.plot(r, q(r, Nmax, 1.4))
    plt.semilogy()
    plt.ylim(top=1,bottom=10**(-6))
plt.show()
