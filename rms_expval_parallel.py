import numpy as np
import matplotlib.pyplot as plt
from create_matrix import mat
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
import time


chi_Nmax150_aHO12 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
chi_Nmax150_aHO14 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
chi_Nmax150_aHO16 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
chi_Nmax150_aHO18 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
chi_Nmax150_aHO20 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
trel_Nmax150 = np.loadtxt(r'Matrixelemente/trel_NMax150_LMax02_aHO01.00.meho', skiprows=1)
rsq_Nmax150 = np.loadtxt(r'Matrixelemente/rsq_NMax150_LMax02.meho', skiprows=1)

chi = np.array([chi_Nmax150_aHO12, chi_Nmax150_aHO14, chi_Nmax150_aHO16, chi_Nmax150_aHO18, chi_Nmax150_aHO20])
aHO = np.array([1.2,1.4,1.6,1.8,2])


# start_time = time.time()

def rms_Nmax(Nmax, j, rsq_Nmax150, aHO_sq, aHO_inv_sq):
    rsq = 0
    h_eigvals, h_eigvecs = np.linalg.eigh(mat(chi[j], Nmax) + mat(trel_Nmax150, Nmax) * aHO_inv_sq)
    gs_index = np.where(h_eigvals == min(h_eigvals))[0][0]
    
    mat_rsq = mat(rsq_Nmax150, Nmax) * aHO_sq/4
    
    rsq_intermediate = np.dot(mat_rsq, h_eigvecs[gs_index])
    rsq = np.dot(h_eigvecs[gs_index], rsq_intermediate)
    rms = np.sqrt(rsq)
    
    return [Nmax, rms]

def rms_aHO(j, Nstep):
    rms = []
    aHO_sq = aHO[j]**2
    aHO_inv_sq = 1 / aHO_sq
    
    Nmax_lst = np.arange(Nstep, 151, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rms_Nmax, Nmax, j, rsq_Nmax150, aHO_sq, aHO_inv_sq) for Nmax in Nmax_lst]
        for future in futures:
            rms.append(future.result())
    
    return rms

rms = np.array(rms_aHO(4, 2))

# stop_time = time.time()

# print((stop_time-start_time)/60)

plt.plot(rms[:, 0],rms[:, 1])


plt.title('$R_{rms}$ bei $a_{HO}$ = 2.0 fm')
plt.ylabel("$R_{rms}$ fm")
plt.xlabel('$N_{max}$')

plt.show()
