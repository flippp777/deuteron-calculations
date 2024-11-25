import numpy as np
import matplotlib.pyplot as plt

Nmax, u, w = np.loadtxt('uw_fitted_kappas_aHO1.2.txt').transpose()
phi0, phi2 = np.loadtxt('phi_fitted_kappas_aHO1.2.txt', usecols=[1, 2]).transpose()

plt.figure()
plt.plot(Nmax, (w-0.2305)/0.2305)
plt.plot(Nmax, (phi2-0.2305)/0.2305)
plt.legend(['w', 'phi2'])
plt.grid()
# plt.yticks([0.2305])

plt.figure()
plt.plot(Nmax, (u-0.2305)/0.2305)
plt.plot(Nmax, (phi0-0.2305)/0.2305)
plt.legend(['u', 'phi0'])
plt.grid()
# plt.yticks([0.2305])

def evolving_average(lst):
    final = []
    for i in range(len(lst)):
        final.append(np.mean(lst[:i]))
    return np.array(final)

plt.figure()
plt.plot(Nmax, evolving_average(w))
plt.plot(Nmax, evolving_average(phi2))
plt.legend(['w', 'phi2'])
plt.grid()
plt.yticks([0.2305])
# 
plt.figure()
plt.plot(Nmax, evolving_average(u))
plt.plot(Nmax, evolving_average(phi0))
plt.legend(['u', 'phi0'])
plt.grid()
plt.yticks([0.2305])

plt.show()