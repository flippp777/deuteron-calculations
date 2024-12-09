import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import quad
import sys

if sys.platform == "win32":
    sys.path.insert(1, r'C:\Users\phili.DESKTOP\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc
from make.matrix import mat
from plotstyle.colorFader import colorFader
from plotstyle.set_size import set_size

plt.style.use('src\stylesheet.mplstyle')


chi_Nmax150_aHO12 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
chi_Nmax150_aHO14 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
chi_Nmax150_aHO16 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
chi_Nmax150_aHO18 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
chi_Nmax150_aHO20 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
trel_Nmax150 = np.loadtxt(r'matrix_elements/trel_NMax150_LMax02_aHO01.00.meho', skiprows=1)
rsq_Nmax150 = np.loadtxt(r'matrix_elements/rsq_NMax150_LMax02.meho', skiprows=1)

chi = np.array([chi_Nmax150_aHO12, chi_Nmax150_aHO14, chi_Nmax150_aHO16, chi_Nmax150_aHO18, chi_Nmax150_aHO20])
aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0])

# print('density integral aHO = 1.2')
# for i in range(0,40,10):
#     print(np.sqrt(quad(lambda r: deuteron_wavefunc(i, 1.2, r)[3] * r**4, 0, np.inf)))
# print('density integral aHO = 1.4')
# for i in range(0,40,10):
#     print(np.sqrt(quad(lambda r: deuteron_wavefunc(i, 1.4, r)[3] * r**4, 0, np.inf)))
# print('density integral aHO = 1.6')
# for i in range(0,40,10):
#     print(np.sqrt(quad(lambda r: deuteron_wavefunc(i, 1.6, r)[3] * r**4, 0, np.inf)))
# print('density integral aHO = 1.8')
# for i in range(0,40,10):
#     print(np.sqrt(quad(lambda r: deuteron_wavefunc(i, 1.8, r)[3] * r**4, 0, np.inf)))
# print('density integral aHO = 2.0')
# for i in range(0,40,10):
#     print(np.sqrt(quad(lambda r: deuteron_wavefunc(i, 2.0, r)[3] * r**4, 0, np.inf)))

def rsq_integral_quad(Nmax, aHO):
    return quad(lambda r: deuteron_wavefunc(Nmax, aHO, r)[3] * r**4, 0, np.inf)[0] / 4 # Warum Faktor 4?

def rsq_integral_trapz(Nmax, aHO, rmax, dr):
    r_size = int(np.round(rmax / dr))
    r = np.linspace(0, rmax + 1, r_size)
    integr = deuteron_wavefunc(Nmax, aHO, r)[3] * r**4
    return np.trapz(integr, r, dr) / 4      # Warum Faktor 4?

def rrms_integral_quad(Nmax, aHO, Nstep):
    rrms = []
    Nmax_lst = np.arange(0, Nmax + 1, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rsq_integral_quad, Nmax, aHO) for Nmax in Nmax_lst]
        for future in futures:
            rrms.append(future.result())
    
    rrms = np.sqrt(np.array(rrms))
    
    return np.array([Nmax_lst, rrms])

def rrms_integral_trapz(Nmax, aHO, Nstep, rmax, dr):
    rrms = []
    Nmax_lst = np.arange(0, Nmax + 1, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rsq_integral_trapz, Nmax, aHO, rmax, dr) for Nmax in Nmax_lst]
        for future in futures:
            rrms.append(future.result())
    
    rrms = np.sqrt(np.array(rrms))
    
    return np.array([Nmax_lst, rrms])


def rrms_Nmax(i, j, rsq_Nmax150, aHO_sq, aHO_inv_sq):
    rsq = 0
    h_eigvals, h_eigvecs = np.linalg.eig(mat(chi[j], i) + mat(trel_Nmax150, i) * aHO_inv_sq)
    gs_index = np.where(h_eigvals == min(h_eigvals))[0][0]
    
    mat_rsq = mat(rsq_Nmax150, i) * aHO_sq/4
    
    rsq = np.dot(h_eigvecs[:, gs_index], np.dot(mat_rsq, h_eigvecs[:, gs_index]))
    rrms = np.sqrt(rsq)
    
    return rrms

def rrms_aHO(j, Nstep):
    rrms = []
    aHO_sq = aHO_lst[j]**2
    aHO_inv_sq = 1 / aHO_sq
    
    Nmax_lst = np.arange(0, 61, Nstep)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rrms_Nmax, i, j, rsq_Nmax150, aHO_sq, aHO_inv_sq) for i in Nmax_lst]
        for future in futures:
            rrms.append(future.result())
    
    return np.array([Nmax_lst, rrms])

rrms_operator = np.array(Parallel(n_jobs=-1)(delayed(rrms_aHO)(j, 1) for j in range(aHO_lst.shape[0])))
Rrms = 1.97
rrms_wavefunc = np.array(Parallel(n_jobs=-1)(delayed(rrms_integral_trapz)(60, aHO, 1, 100, 0.1) for aHO in aHO_lst))

# print((np.mean(rrms_wavefunc[:,1,150]), np.std(rrms_wavefunc[:,1,150])))
# print((np.mean(rrms_wavefunc[:,1,60]), np.std(rrms_wavefunc[:,1,60])))


plt.figure(figsize=set_size(502.14963))
for i in range(aHO_lst.shape[0]):
    plot = plt.plot(rrms_operator[i, 0], rrms_operator[i, 1], 'x-', color=colorFader('#0000FF', '#00004c', i/len(aHO_lst)), markersize=4)
    plt.plot(rrms_wavefunc[i, 0], rrms_wavefunc[i, 1], 'x', linestyle='-.', color=colorFader('#FF0000', '#4c0000', i/len(aHO_lst)), markersize=4)
legend = np.array([['\(a_\\text{HO}=\ \)' + f'\({aHO}\ \)' +'fm, Operator', '\(a_\\text{HO}=\ \)' + f'\({aHO}\ \)' +'fm, Wave function'] for aHO in aHO_lst]).flatten()
plt.legend(legend, loc='lower right', ncol=2)
plt.ylabel("$R_\\text{rms}$ [fm]")
plt.yticks([1.2, 1.4, 1.6, 1.8, Rrms, 2.0, 2.2], ['1.20', '1.40', '1.60', '1.80', '1.97', '', '2.20'])
plt.xlabel('$\\Tilde{N}_\\text{max}$')
plt.xlim(0, 60)
plt.grid(linestyle='--')

plt.savefig('rrms_Nmax60.pdf', bbox_inches='tight')

plt.show()