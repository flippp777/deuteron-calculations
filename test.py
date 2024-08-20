import numpy as np
from create_matrix import mat

matrix = np.diag([1, 2, 3])
eigvals, eigvecs = np.linalg.eigh(matrix)
print(eigvecs[:,2])

