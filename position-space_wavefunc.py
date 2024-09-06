import numpy as np
from matplotlib import pyplot as plt
from scipy.special import genlaguerre, gamma
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
    
    return np.exp(-0.5 * r**2) * r**L * genlaguerre(N, L)(r) / aHO**(3/2)

def pos_wavefunc_norm(N, L):
    return np.sqrt(2 * gamma(N + 1) / gamma(N + L + 1.5))

def create_deuteron_wavefunc(Nmax, aHO):
    def deuteron_wavefunc(r):
        gs_vec = gs_eigvec(Nmax, aHO)
         
        deut_wavefunc = 0
        for N in range(Nmax + 1):
            deut_wavefunc += gs_vec[N] * pos_wavefunc(N, 0, aHO, r) / pos_wavefunc_norm(N, 0)
        for N in range(Nmax + 2, (Nmax+1) * 2):
            deut_wavefunc += gs_vec[N] * pos_wavefunc(N, 2, aHO, r) / pos_wavefunc_norm(N, 2)
        return deut_wavefunc
    return deuteron_wavefunc
        
test_func = create_deuteron_wavefunc(40, 1.6)

x = np.linspace(0, 10, 100)
y = test_func(x)

plt.plot(x, y)
plt.show()
