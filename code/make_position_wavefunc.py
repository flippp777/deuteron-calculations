import numpy as np
from matplotlib import pyplot as plt
from scipy.special import eval_genlaguerre, gammaln
from make_matrix import mat
from sys import exit


chi_Nmax150_aHO12 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
chi_Nmax150_aHO14 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
chi_Nmax150_aHO16 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
chi_Nmax150_aHO18 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
chi_Nmax150_aHO20 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
trel_Nmax150 = np.loadtxt(r'matrix_elements/trel_NMax150_LMax02_aHO01.00.meho', skiprows=1)


def gs_eigvec(Nmax, aHO):
    if aHO == 1.2:
        chi = chi_Nmax150_aHO12
    elif aHO == 1.4:
        chi = chi_Nmax150_aHO14
    elif aHO == 1.6:
        chi = chi_Nmax150_aHO16
    elif aHO == 1.8:
        chi = chi_Nmax150_aHO18
    elif aHO == 2.0:
        chi = chi_Nmax150_aHO20
    elif aHO == 2:
        chi = chi_Nmax150_aHO20
    else:
        exit('! aHO = ' + str(aHO) + ' is not supported!')
        
    h_eigvals, h_eigvecs = np.linalg.eig(mat(chi, Nmax) + mat(trel_Nmax150, Nmax) * 1/aHO**2)
    gs_index = np.where(h_eigvals == min(h_eigvals))[0][0]
    
    return h_eigvecs[:, gs_index]


def pos_wavefunc(N, L, aHO, r):
    r = r/aHO
    return  np.exp(-0.5 * r**2) * r**L * eval_genlaguerre(N, L + 0.5, r**2) / (aHO * np.sqrt(aHO))


def pos_wavefunc_norm(N, L):
    return np.sqrt(2) * np.exp(0.5 * (gammaln(N + 1) - gammaln(N + L + 1.5)))


def deuteron_wavefunc(Nmax, aHO, r):
    gs_vec = gs_eigvec(Nmax, aHO)
        
    deut_wavefunc_L0 = 0
    deut_wavefunc_L2 = 0
    
    for N in range(Nmax + 1):
        deut_wavefunc_L0 += gs_vec[N] * pos_wavefunc(N, 0, aHO, r) * pos_wavefunc_norm(N, 0)
    for N in range(Nmax + 1):
        deut_wavefunc_L2 += gs_vec[N + Nmax +1] * pos_wavefunc(N, 2, aHO, r) * pos_wavefunc_norm(N, 2)
    
    deut_wavefunc = deut_wavefunc_L0 + deut_wavefunc_L2
    deut_density = deut_wavefunc_L0**2 + deut_wavefunc_L2**2
    
    return [deut_wavefunc, deut_wavefunc_L0, deut_wavefunc_L2, deut_density]