import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit

sys.path.insert(1, r'C:\Users\Phili\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')

from make.position_space_wavefunc import deuteron_wavefunc
from rrms_density_continnuation_exp.rrms_density_continuation_overlap_Nmax_minus_1 import find_cont_par_cut

plt.style.use('tableau-colorblind10')

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
    
def derivative_model(Nmax, aHO, kappa, xmax):
    r = np.arange(0, 100, 0.1)
    h0 = (1/(kappa * r) * np.exp(-kappa * r))
    h2 = (1/(kappa * r) * np.exp(-kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2))
    plt.figure()
    plt.title('d/d$r$ ln$(|\psi(r)|^2)$ für $N_{max} = $' + str(Nmax) + ' und $a_{HO} = $' + str(aHO) + ' fm und d/d$r$ ln$(|h_l(i\kappa r)|^2)$ für $\kappa = $' +str(kappa))
    plt.plot(r, derivative(r, np.log(deut_density(r, Nmax, aHO)), 1))
    plt.plot(r, derivative(r, np.log(h0**2), 1))
    plt.plot(r, derivative(r, np.log(h2**2), 1))
    plt.plot(r, derivative(r, np.log((h0 + h2)**2), 1))
    plt.legend(['d/d$r$ ln$(|\psi(r)|^2)$' ,'d/d$r$ ln$(|h_0(i\kappa r)|^2)$', 'd/d$r$ ln$(|h_2(i\kappa r)|^2)$', 'd/d$r$ ln$(|h_0(i\kappa r) + h_2(i\kappa r)|^2)$'])
    plt.ylim(-1.4, -0.4)
    plt.xlim(0, xmax)
    plt.grid()
    
def fit_derivative(Nmax, aHO, rmax):
    r_fit = np.arange(2.5, rmax + 0.1, 0.1)
    def lnh0_model(r, kappa):
        return -2 * (kappa + 1/r)

    def lnh0h2_model(r, kappa):
        return - (6/4 - (2 * kappa**2 *r)/(1 + kappa * r))
    
    lnpsi = derivative(r_fit, np.log(deut_density(r_fit, Nmax, aHO)), 1) 
    
    kappa_h0 = curve_fit(lnh0_model, r_fit, lnpsi, 0.2305)[0][0]
    kappa_h0h2 = curve_fit(lnh0h2_model, r_fit, lnpsi, 0.2305)[0][0]
    
    plt.figure()
    plt.title('d/d$r$ ln$(|\psi(r)|^2)$ für $N_{max} = $' + str(Nmax) + ' und $a_{HO} = $' + str(aHO) + ' fm mit gefitteten d/d$r$ ln$(|h_l(i\kappa r)|^2)$')
    plt.plot(r_fit, lnpsi)
    plt.plot(r_fit, lnh0_model(r_fit, kappa_h0))
    plt.plot(r_fit, lnh0h2_model(r_fit, kappa_h0h2))
    plt.ylim(-1.4, -0.45)
    plt.legend(['d/d$r$ ln$(|\psi(r)|^2)$' ,'d/d$r$ ln$(|h_0(i\kappa r)|^2)$, $\kappa = $' + str(kappa_h0) + ' 1/fm', 'd/d$r$ ln$(|h_0(i\kappa r) + h_2(i\kappa r)|^2)$, $\kappa = $' + str(kappa_h0h2) + ' 1/fm'])
    plt.grid()

aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
       
# # slope_at_r(20)
# slope_at_Nmax(150, 20, 10, 1.2, 0, 34)
# slope_at_Nmax(150, 20, 10, 1.4, 0, 40)
# slope_at_Nmax(150, 20, 10, 1.6, 0, 47)
# slope_at_Nmax(150, 20, 10, 1.8, 0, 54)
# slope_at_Nmax(150, 20, 10, 2.0, 0, 60)

derivative_model(3, 1.2, 0.2305, 10)
fit_derivative(10, 1.2, 8)
plt.show()