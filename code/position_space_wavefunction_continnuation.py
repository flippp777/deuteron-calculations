import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import quad
from scipy.optimize import curve_fit

from make_position_space_wavefunc import deuteron_wavefunc

def lin_cont_4(Nmax, aHO):
    def lin_log_model(x, a, b):
        return x * a + b

    r_fit = [3.9, 4]
    logy_fit = np.log(deuteron_wavefunc(Nmax, aHO, r_fit)[3])
    popt = curve_fit(lin_log_model, r_fit, logy_fit)[0]
    
    r = np.linspace(4,15, 10)
    y = lin_log_model(r, popt[0], popt[1])

    return np.array([r, y])

aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
Nmax_lst = np.arange(0, 11, 5)
r = np.linspace(0, 15, 500)

for aHO in aHO_lst:
    plt.figure()
    legend = []
    for Nmax in Nmax_lst:
        cont = lin_cont_4(Nmax, aHO)
        plt.semilogy(r, deuteron_wavefunc(Nmax, aHO, r)[3])
        plt.semilogy(cont[0], np.exp(cont[1]))
        legend.append('$N_{max} = $' + str(Nmax) + ', Probabiliy densitz')
        legend.append('$N_{max} = $' + str(Nmax) + ', Continuation')
        
    plt.ylim(bottom=10**(-6))
    plt.legend(legend)
    
    
plt.show()