import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.special
from create_matrix import mat
from sys import exit


chi_Nmax150_aHO12 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
chi_Nmax150_aHO14 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
chi_Nmax150_aHO16 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
chi_Nmax150_aHO18 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
chi_Nmax150_aHO20 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
trel_Nmax150 = np.loadtxt(r'Matrixelemente/trel_NMax150_LMax02_aHO01.00.meho', skiprows=1)

chi = np.array([chi_Nmax150_aHO12, chi_Nmax150_aHO14, chi_Nmax150_aHO16, chi_Nmax150_aHO18, chi_Nmax150_aHO20])
aHO = np.array([1.2,1.4,1.6,1.8,2])

def gs_eigvec(Nmax, aHO):
    if aHO == 1.2:
        chi = chi_Nmax150_aHO12
    elif aHO == 1.4:
        chi =chi_Nmax150_aHO14
    elif aHO == 1.6:
        chi =chi_Nmax150_aHO16
    elif aHO == 1.8:
        chi =chi_Nmax150_aHO18
    elif aHO == 2.0:
        chi =chi_Nmax150_aHO20
    elif aHO == 2:
        chi =chi_Nmax150_aHO20
    else:
        exit('! aHO = ' + str(aHO) + ' is not supported!')
        
    h_eigvals, h_eigvecs = np.linalg.eig(mat(chi, Nmax) + mat(trel_Nmax150, Nmax) * 1/aHO**2)
    gs_index = np.where(h_eigvals == min(h_eigvals))[0][0]
    
    return h_eigvecs[:, gs_index]

def pos_wavefunc(N, L, aHO, r):
    r = r/aHO
    return np.exp(-0.5 * r**2) * r**L * scipy.special.genlaguerre(N, L)(r) / aHO**(3/2)

def pos_wavefunc_norm(N, L):
    return np.sqrt(2 * scipy.special.gamma(N + 1) / scipy.special.gamma(N + L + 1.5))

print(gs_eigvec(10, 1.2))
print(pos_wavefunc(1,1,1.2,1))
print(pos_wavefunc_norm(1,1))