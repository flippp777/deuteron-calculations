import numpy as np
from create_matrix import mat

rsq_Nmax20 = np.loadtxt(r'Matrixelemente/rsq_NMax20_LMax02.meho', skiprows=1)

con_SJTMT = (
        (rsq_Nmax20[:,0] <= 4) & # N <= Nmax
        (rsq_Nmax20[:,2] <= 4) & # N' <= Nmax
        (rsq_Nmax20[:, 4] == 1) &  # S=1
        (rsq_Nmax20[:, 5] == 1) &  # J=1
        (rsq_Nmax20[:, 6] == 0) &  # T=0
        (rsq_Nmax20[:, 7] == 0))    # MT=0
    
c_rsq_Nmax20 = rsq_Nmax20[con_SJTMT]

print(np.round(c_rsq_Nmax20, decimals=3))
np.savetxt('rsq_Nmax4_test.txt', np.round(c_rsq_Nmax20, decimals=3))

print(np.round(mat(rsq_Nmax20, 4), decimals=3))
np.savetxt('rsq_Nmax4_test_mat.txt', np.round(mat(rsq_Nmax20, 4), decimals=3))