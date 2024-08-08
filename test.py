import numpy as np

rsq_Nmax150_lin = np.loadtxt(r'Matrixelemente/rsq_NMax150_LMax02.meho', skiprows=1)
rsq_Nmax150_win = np.loadtxt(r'Matrixelemente\rsq_NMax150_LMax02.meho', skiprows=1)
print(rsq_Nmax150_lin)
print(rsq_Nmax150_win)