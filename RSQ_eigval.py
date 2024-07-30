import numpy as np
import matplotlib.pyplot as plt
from create_matrix import mat

rsq_Nmax150 = np.loadtxt(r'Matrixelemente/rsq_NMax150_LMax02 (1).meho', skiprows=1)

Nmax_step2 = np.arange(2,152,2)
rsq = np.array([])

for i in Nmax_step2:
    rsq = np.append(rsq, [min(np.linalg.eig(mat(rsq_Nmax150, i))[0])])

print(rsq[-1])
plt.scatter(Nmax_step2,rsq)
plt.show()