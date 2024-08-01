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

start = time.time()

Nmax_lst = np.arange(50, 151, 50)

def rsq_n(vec, mat_rsq, aHO_j_sq):
    return np.dot(vec, np.dot(mat_rsq * aHO_j_sq, vec))

def compute_rms_j(j):
    rms_j = []
    aHO_j_sq = aHO[j]**2
    aHO_j_inv_sq = 1 / aHO_j_sq
    
    for i in Nmax_lst:
        rsq = 0
        h_eigvecs = np.linalg.eig(mat(chi[j], i) + mat(trel_Nmax150, i) * aHO_j_inv_sq)[1]
        
        mat_rsq = mat(rsq_Nmax150, i)
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(rsq_n, vec, mat_rsq, aHO_j_sq) for vec in h_eigvecs]
            for future in futures:
                rsq += future.result()
        
        rms_j.append(np.sqrt(rsq))
    
    return rms_j

rms = np.array(Parallel(n_jobs=-1)(delayed(compute_rms_j)(j) for j in range(aHO.shape[0])))

print(rms)
stop = time.time()
print(stop - start)
