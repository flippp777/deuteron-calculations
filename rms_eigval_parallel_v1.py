import numpy as np
import matplotlib.pyplot as plt
from create_matrix import mat
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
import time


chi_Nmax150_aHO12 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
chi_Nmax150_aHO14 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
chi_Nmax150_aHO16 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
chi_Nmax150_aHO18 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
chi_Nmax150_aHO20 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
trel_Nmax150 = np.loadtxt(r'Matrixelemente\trel_NMax150_LMax02.meho', skiprows=1)
rsq_Nmax150 = np.loadtxt(r'Matrixelemente/rsq_NMax150_LMax02.meho', skiprows=1)

chi = np.array([chi_Nmax150_aHO12, chi_Nmax150_aHO14, chi_Nmax150_aHO16, chi_Nmax150_aHO18, chi_Nmax150_aHO20])
aHO = np.array([1.2, 1.4, 1.6, 1.8, 2.0])


Nmax_lst = np.arange(5, 151, 5)

def rsq_n(vec, mat_rsq, aHO_sq):
    return np.dot(vec, np.dot(mat_rsq * aHO_sq, vec))

def compute_rms(j):
    rms = []
    aHO_sq = aHO[j]**2
    aHO_inv_sq = 1 / aHO_sq
    
    for i in Nmax_lst:
        rsq = 0
        h_eigvecs = np.linalg.eig(mat(chi[j], i) + mat(trel_Nmax150, i) * aHO_inv_sq)[1]
        
        mat_rsq = mat(rsq_Nmax150, i)
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(rsq_n, vec, mat_rsq, aHO_sq) for vec in h_eigvecs]
            for future in futures:
                rsq += future.result()
        
        rms.append(np.sqrt(rsq))
    
    return rms

rms = np.array(Parallel(n_jobs=-1)(delayed(compute_rms)(j) for j in range(aHO.shape[0])))

for i in range(rms.shape[0]):
    plt.scatter(Nmax_lst,rms[i])

legend = np.array([])
for i in aHO:
    legend = np.append(legend,'$a_{HO}$ = '+ str(i) + ' fm')

plt.title('$R_{rms}$ bei verschiedenen $a_{HO}$')
plt.legend(legend)
plt.ylabel("$R_{rms}$ fm")
plt.xlabel('$N_{max}$')

plt.show()
