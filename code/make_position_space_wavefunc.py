import numpy as np
from matplotlib import pyplot as plt
from scipy.special import eval_genlaguerre, gammaln, gamma
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
        exit('!aHO = ' + str(aHO) + ' is not supported!')
        
    h_eigvals, h_eigvecs = np.linalg.eig(mat(chi, Nmax) + mat(trel_Nmax150, Nmax) * 1/aHO**2)
    gs_index = np.where(h_eigvals == min(h_eigvals))[0][0]

    return h_eigvecs[:, gs_index]


def pos_wavefunc(N, L, aHO, r):
    aHOrel = np.sqrt(2) * aHO
    r = r/aHOrel
    return  np.exp(-0.5 * r**2) * r**L * eval_genlaguerre(N, L + 0.5, r**2) / (aHOrel * np.sqrt(aHOrel))


def pos_wavefunc_norm(N, L):
    return np.sqrt(2) * np.exp(0.5 * (gammaln(N + 1) - gammaln(N + L + 1.5)))
    # return np.sqrt(2 * gamma(N+1) / gamma(N + L + 1.5))

plt.figure()
for i in range(40,50):
    plt.plot(pos_wavefunc(i,0, 1.8, np.linspace(0, 10, 100)) * pos_wavefunc_norm(i, 0))
plt.legend([i for i in range(10)])

def deuteron_wavefunc(Nmax, aHO, r):
    # print('Nmax = ' + str(Nmax))
    gs_vec = gs_eigvec(Nmax, aHO)
    # print('dim(gs_eigvec) = ' + str(gs_vec.shape))
    deut_wavefunc_L0 = 0
    deut_wavefunc_L2 = 0
    
    if Nmax == 0:
        deut_wavefunc_L0 = gs_vec[Nmax] * pos_wavefunc(Nmax, 0, aHO, r) * pos_wavefunc_norm(Nmax, 0)
        # print('Nmax == 0 case, L=0: Nmax = ' + str(Nmax))
        deut_wavefunc_L2 = gs_vec[1 + Nmax] * pos_wavefunc(Nmax, 2, aHO, r) * pos_wavefunc_norm(Nmax, 2)
        # print('Nmax == 0 case, L=2: Nmax + 1 = ' + str(Nmax +1))
    else:
        plt.figure()
        for N in range(Nmax + 1):
            deut_wavefunc_L0 += gs_vec[N] * pos_wavefunc(N, 0, aHO, r) * pos_wavefunc_norm(N, 0)
            # print('L = 0, N = ' + str(N))
            plt.plot(r, deut_wavefunc_L0)
        for N in range(Nmax + 1):
            deut_wavefunc_L2 += gs_vec[N + Nmax + 1] * pos_wavefunc(N, 2, aHO, r) * pos_wavefunc_norm(N, 2)
            # print('L = 2, N = ' + str(N))
            # print('L = 2, N + Nmax + 1= ' + str(N + Nmax + 1))
        plt.legend([i for i in range(Nmax+1)])
        
    deut_wavefunc = deut_wavefunc_L0 + deut_wavefunc_L2
    deut_density = deut_wavefunc_L0**2 + deut_wavefunc_L2**2
    
    return [deut_wavefunc, deut_wavefunc_L0, deut_wavefunc_L2, deut_density]

# Alternative way to calculate the the wavefunction at different r with one r at a time. 
# It is the same as before 
def deuteron_wavefunc_single_r(Nmax, aHO, r):
    gs_vec = gs_eigvec(Nmax, aHO)
        
    deut_wavefuncs = np.array([[0,0],[0,0]])
    for x in r:
        deut_wavefunc_L0 = 0
        deut_wavefunc_L2 = 0
        for N in range(Nmax + 1):
            deut_wavefunc_L0 += gs_vec[N] * pos_wavefunc(N, 0, aHO, x) * pos_wavefunc_norm(N, 0)
        for N in range(Nmax + 1):
            deut_wavefunc_L2 += gs_vec[N + Nmax + 1] * pos_wavefunc(N, 2, aHO, x) * pos_wavefunc_norm(N, 2)
        deut_wavefuncs = np.append(deut_wavefuncs, [[deut_wavefunc_L0,deut_wavefunc_L2]], axis=0)
    deut_wavefuncs = np.delete(deut_wavefuncs, (0,1), axis=0)

    return deut_wavefuncs 

# print(deuteron_wavefunc(0, 1.8, np.array([0,1])))
# print(deuteron_wavefunc(9, 1.8, np.linspace(0,10, 100)))
# for i in range(0, 91, 10):
    # print(np.dot(gs_eigvec(i, 1.2), gs_eigvec(i, 1.2)))
# plt.show()