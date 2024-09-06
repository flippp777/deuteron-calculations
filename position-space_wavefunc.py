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

# chi = np.array([chi_Nmax150_aHO12, chi_Nmax150_aHO14, chi_Nmax150_aHO16, chi_Nmax150_aHO18, chi_Nmax150_aHO20])
# aHO = np.array([1.2,1.4,1.6,1.8,2])

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
    
    return np.exp(-0.5 * r**2) * r**L * genlaguerre(N, L + 0.5)(r**2) / aHO**(3/2)

def pos_wavefunc_norm(N, L):
    return np.sqrt(2 * gamma(N + 1) / gamma(N + L + 1.5))

def create_deuteron_wavefunc(Nmax, aHO, r):
    # def deuteron_wavefunc(r):
    gs_vec = gs_eigvec(Nmax, aHO)
        
    deut_wavefunc_L0 = 0
    deut_wavefunc_L2 = 0
    for N in range(Nmax + 1):
        deut_wavefunc_L0 += gs_vec[N] * pos_wavefunc(N, 0, aHO, r) * pos_wavefunc_norm(N, 0)
    for N in range(Nmax + 1, (Nmax+1) * 2 ):
        deut_wavefunc_L2 += gs_vec[N] * pos_wavefunc(N, 2, aHO, r) * pos_wavefunc_norm(N, 2)
    deut_wavefunc = deut_wavefunc_L0 + deut_wavefunc_L2
    
    return [deut_wavefunc, deut_wavefunc_L0, deut_wavefunc_L2]
    # return deuteron_wavefunc
        
# wavefunc = create_deuteron_wavefunc(60, 1.4)


plt.figure()
x_model = np.linspace(0, 10, 500)
wavefunc = create_deuteron_wavefunc(60, 1.4, x_model)
wavefunc_model, wavefunc_L0_model, wavefunc_L2_model = wavefunc[0], wavefunc[1], wavefunc[2]

plt.plot(x_model, wavefunc_model)
plt.plot(x_model, wavefunc_L0_model)
plt.plot(x_model, wavefunc_L2_model)
plt.legend(['Wavefunction', 'L=0', 'L=2'])
plt.xlabel('r/aHO (fm)')
plt.show()
