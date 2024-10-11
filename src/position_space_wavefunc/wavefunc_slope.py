import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, r'C:\Users\Phili\OneDrive\Uni OneDrive\Bachelorthesis\deuteron-calculations\src')

from make.position_space_wavefunc import deuteron_wavefunc
from rrms_density_continnuation.rrms_density_continuation_overlap_Nmax_minus_1 import find_cont_par_cut


def deut_density(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[3]

def derivative(x, f, n):
    dx = abs(x[0] - x[1])
    
    for i in range(n):
        f = np.gradient(f) / dx
        
    return f

def slope_at_r(r):
    r_space = [r - 10**(-6), r, r + 10**(-6)]
    Nmax_lst = np.arange(0, 65, 5)
    for aHO in aHO_lst:
        plt.figure()
        plt.title("Steigung der logarithmischen Aufnethaltwahrscheinlichkeit an der Stelle r = " + str(r) + " (fm) für $a_{HO} = $" + str(aHO) + " (fm)")
        slope = []
        slope_fit = []
        for Nmax in Nmax_lst:
            slope.append(derivative(r_space, np.log(deut_density(r_space, Nmax, aHO)), 1)[1])
            slope_fit.append(find_cont_par_cut(Nmax, aHO)[0])
        plt.plot(Nmax_lst, slope_fit)
        plt.plot(Nmax_lst, slope)
        plt.legend(['Fit', 'Derivative'])
        plt.grid()
        print(aHO, slope[-1], slope_fit[-1])

def slope_at_Nmax(NNmax, NNmin, dNmax, aHO, rmin, rmax):
    r = np.arange(rmin, rmax+0.1, 0.1)
    Nmax_lst = np.arange(NNmin, NNmax, dNmax)
    plt.figure()
    for Nmax in Nmax_lst:
        plt.title("Steigung der logarithmischen Aufnethaltwahrscheinlichkeit für $a_{HO}$ = " + str(aHO) + ' fm und verschiedene $N_{max}$')
        plt.plot(r, derivative(r, np.log(deut_density(r, Nmax, aHO)), 1))
        plt.ylim(-1.4,-0.5)
        # plt.plot(r, np.log(deut_density(r, Nmax, aHO)))
        plt.grid()
    plt.legend(['$N_{max} = $' + str(Nmax) for Nmax in Nmax_lst])
    

aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
       
# slope_at_r(20)
slope_at_Nmax(150, 20, 10, 1.2, 0, 34)
slope_at_Nmax(150, 20, 10, 1.4, 0, 40)
slope_at_Nmax(150, 20, 10, 1.6, 0, 47)
slope_at_Nmax(150, 20, 10, 1.8, 0, 54)
slope_at_Nmax(150, 20, 10, 2.0, 0, 60)

plt.show()