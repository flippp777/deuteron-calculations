import numpy as np
import matplotlib.pyplot as plt
from create_matrix import mat
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
import time

chi_Nmax150_aHO12 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
chi_Nmax150_aHO14 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
chi_Nmax150_aHO16 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
chi_Nmax150_aHO18 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
chi_Nmax150_aHO20 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
trel_Nmax150 = np.loadtxt(r'Matrixelemente\trel_NMax150_LMax02.meho', skiprows=1)
rsq_Nmax150 = np.loadtxt(r'Matrixelemente/rsq_NMax150_LMax02.meho', skiprows=1)

chi = np.array([chi_Nmax150_aHO12, chi_Nmax150_aHO14, chi_Nmax150_aHO16, chi_Nmax150_aHO18, chi_Nmax150_aHO20])
aHO = np.array([1.2,1.4,1.6,1.8,2])

chi_mat = mat(chi_Nmax150_aHO16, 100)
trel_mat = mat(trel_Nmax150, 100) * 1/1.6**2
rsq_mat = mat(rsq_Nmax150, 100) * 1.6**2

h_eigval, h_eigvec = np.linalg.eig(chi_mat + trel_mat)
gs_index = np.where(h_eigval == min(h_eigval))[0][0]


rsq = np.dot(h_eigvec[gs_index], np.dot(rsq_mat, h_eigvec[gs_index]))


# start_time = time.time()

def rms_Nmax(i, j, rsq_Nmax150, aHO_sq, aHO_inv_sq):
    rsq = 0
    h_eigvals, h_eigvecs = np.linalg.eig(mat(chi[j], i) + mat(trel_Nmax150, i) * aHO_inv_sq)
    gz_index = np.where(h_eigvals == min(h_eigvals))[0][0]
    
    mat_rsq = mat(rsq_Nmax150, i)
    
    rsq = np.dot(h_eigvecs[gz_index], np.dot(mat_rsq * aHO_sq, h_eigvecs[gz_index]))
    rms = np.sqrt(rsq)
    
    return [i, rms]

def rms_aHO(j, Nstep):
    rms = []
    aHO_sq = aHO[j]**2
    aHO_inv_sq = 1 / aHO_sq
    
    Nmax_lst = np.arange(Nstep, 151, Nstep)
    
    for i in Nmax_lst:
            rms.append(rms_Nmax(i, j, rsq_Nmax150, aHO_sq, aHO_inv_sq))
    
    return rms

rms = []
for j in range(aHO.shape[0]):
    rms.append(rms_aHO(j,1))
rms = np.array(rms)

# stop_time = time.time()

# print(stop_time-start_time)

for i in range(rms.shape[0]):
    plt.plot(rms[i,::, 0],rms[i,::, 1])

legend = np.array([])
for i in aHO:
    legend = np.append(legend,'$a_{HO}$ = '+ str(i) + ' fm')

plt.title('$R_{rms}$ bei verschiedenen $a_{HO}$')
plt.legend(legend)
plt.ylabel("$R_{rms}$ fm")
plt.xlabel('$N_{max}$')

plt.show()