import numpy as np
from matplotlib import pyplot as plt
from scipy.special import eval_genlaguerre, gammaln
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
    
    return np.exp(-0.5 * r**2) * r**L * eval_genlaguerre(N, L + 0.5, r**2) / (aHO * np.sqrt(aHO)) 

def pos_wavefunc_norm(N, L):
    return np.sqrt(2) * np.exp(0.5 * (gammaln(N + 1) - gammaln(N + L + 1.5)))

def deuteron_wavefunc(Nmax, aHO, r):
    gs_vec = gs_eigvec(Nmax, aHO)
    print(gs_vec)
        
    deut_wavefunc_L0 = 0
    deut_wavefunc_L2 = 0
    
    for N in range(Nmax + 1):
        deut_wavefunc_L0 += gs_vec[N] * pos_wavefunc(N, 0, aHO, r) * pos_wavefunc_norm(N, 0)
        # print([0, N, gs_vec[N], pos_wavefunc(N, 0, aHO, r), pos_wavefunc_norm(N, 0), deut_wavefunc_L0])
    for N in range(Nmax + 1, (Nmax+1) * 2 ):
        print(N)
        deut_wavefunc_L2 += gs_vec[N] * pos_wavefunc(N, 2, aHO, r) * pos_wavefunc_norm(N, 2)
        # print([2, N, gs_vec[N], pos_wavefunc(N, 2, aHO, r), pos_wavefunc_norm(N, 2), deut_wavefunc_L2])
    
    deut_wavefunc = deut_wavefunc_L0 + deut_wavefunc_L2
    
    return [deut_wavefunc, deut_wavefunc_L0, deut_wavefunc_L2]


plt.figure()

x_model = np.linspace(0, 15, 500)
x_model_log = np.linspace(0, 30, 10)
wavefunc = deuteron_wavefunc(40, 1.4, x_model)
wavefunc_model, wavefunc_L0_model, wavefunc_L2_model = wavefunc[0], wavefunc[1], wavefunc[2]
# wavefunc_log = deuteron_wavefunc(15, 1.4, x_model_log)
# wavefunc_model_log, wavefunc_L0_model_log, wavefunc_L2_model_log = wavefunc_log[0], wavefunc_log[1], wavefunc_log[2]

# print(np.sum(wavefunc_L0_model**2))

# plt.subplot(221)
plt.plot(x_model, wavefunc_model)
plt.plot(x_model, wavefunc_L0_model)
plt.plot(x_model, wavefunc_L2_model)
plt.title('Wellenfunctionen bei $a_{HO} = 1.4$ für $N_{max} = 15$')
plt.legend(['$\Phi(r)$', '$\Phi_{L=0}(r)$', '$\Phi_{L=2}(r)$'])
plt.xlabel('r/aHO (fm)')



# plt.subplot(222)

# plt.semilogy(x_model_log, wavefunc_model_log)
# plt.semilogy(x_model_log, wavefunc_L0_model_log)
# plt.semilogy(x_model_log, wavefunc_L2_model_log)
# plt.title('Wellenfunctionen bei $a_{HO} = 1.4$ für $N_{max} = 15$ mit logarithmischer x-Achse')
# plt.legend(['$\Phi(r)$', '$\Phi_{L=0}(r)$', '$\Phi_{L=2}(r)$'])
# plt.xlabel('r/aHO (fm)')

# wavefunc = deuteron_wavefunc(60, 1.4, x_model)
# wavefunc_model, wavefunc_L0_model, wavefunc_L2_model = wavefunc[0], wavefunc[1], wavefunc[2]
# wavefunc_log = deuteron_wavefunc(60, 1.4, x_model_log)
# wavefunc_model_log, wavefunc_L0_model_log, wavefunc_L2_model_log = wavefunc_log[0], wavefunc_log[1], wavefunc_log[2]

# plt.subplot(223)
# plt.plot(x_model, wavefunc_model)
# plt.plot(x_model, wavefunc_L0_model)
# plt.plot(x_model, wavefunc_L2_model)
# plt.title('Wellenfunctionen bei $a_{HO} = 1.4$ für $N_{max} = 60$')
# plt.legend(['$\Phi(r)$', '$\Phi_{L=0}(r)$', '$\Phi_{L=2}(r)$'])
# plt.xlabel('r/aHO (fm)')


# plt.subplot(224)
# plt.semilogy(x_model_log, wavefunc_model_log)
# plt.semilogy(x_model_log, wavefunc_L0_model_log)
# plt.semilogy(x_model_log, wavefunc_L2_model_log)
# plt.title('Wellenfunctionen bei $a_{HO} = 1.4$ für $N_{max} = 60$ mit logarithmischer x-Achse')
# plt.legend(['$\Phi(r)$', '$\Phi_{L=0}(r)$', '$\Phi_{L=2}(r)$'])
# plt.xlabel('r/aHO (fm)')

# plt.figure()

# plt.subplot(211)
# plt.plot(x_model, pos_wavefunc(1, 2, 1.4, x_model) * pos_wavefunc_norm(1, 2)) 

# plt.subplot(212)
# plt.plot(x_model, pos_wavefunc(20, 2, 1.4, x_model) * pos_wavefunc_norm(20, 2))

plt.show()
