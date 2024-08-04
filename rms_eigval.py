import numpy as np
import matplotlib.pyplot as plt
from create_matrix import mat
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

chi_mat = mat(chi_Nmax150_aHO16, 50)
trel_mat = mat(trel_Nmax150, 50) * 1/1.6**2
rsq_mat = mat(rsq_Nmax150, 50) * 1.6**2


h_eigval, h_eigvec = np.linalg.eig(chi_mat + trel_mat)
rsq = 0
for h_vec in h_eigvec:
    rsq += np.dot(h_vec, np.dot(rsq_mat, h_vec))
print(np.sqrt(rsq))


plt.show()