import numpy as np

def mat(mat_el, Nmax):
    #Conditions für den Deuterium GZ und N, N' truncation
    con_SJTMT = (
        (mat_el[:,0] <= Nmax) & # N <= Nmax
        (mat_el[:,2] <= Nmax) & # N' <= Nmax
        (mat_el[:, 4] == 1) &  # S=1
        (mat_el[:, 5] == 1) &  # J=1
        (mat_el[:, 6] == 0) &  # T=0
        (mat_el[:, 7] == 0))    # MT=0
    
    c_mat_el = mat_el[con_SJTMT]
    
    # Aufteilung der Matrixelemente anhand von L und L'
    L0_L0 = c_mat_el[((c_mat_el[:, 1] == 0) & (c_mat_el[:, 3] == 0))]
    L0_L2 = c_mat_el[((c_mat_el[:, 1] == 0) & (c_mat_el[:, 3] == 2))]
    L2_L0 = c_mat_el[((c_mat_el[:, 1] == 2) & (c_mat_el[:, 3] == 0))]
    L2_L2 = c_mat_el[((c_mat_el[:, 1] == 2) & (c_mat_el[:, 3] == 2))]
    
    # Quadratische L=0, L'=0 Matrix
    q_L0_L0 = np.zeros((Nmax+1, Nmax+1))
    for i in range(L0_L0.shape[0]):
        q_L0_L0[L0_L0[i,2].astype(int), L0_L0[i,0].astype(int)] = L0_L0[i,8]
        q_L0_L0[L0_L0[i,0].astype(int), L0_L0[i,2].astype(int)] = L0_L0[i,8]
        
    # Quadratische L=0, L'=2 Matrix
    q_L0_L2 = np.zeros((Nmax+1, Nmax+1))
    for i in range(L0_L2.shape[0]):
        q_L0_L2[L0_L2[i,0].astype(int), L0_L2[i,2].astype(int)] = L0_L2[i,8]
        q_L0_L2[L2_L0[i,2].astype(int), L2_L0[i,0].astype(int)] = L2_L0[i,8]
    
    # Quadratische L=2, L'=0 Matrix
    q_L2_L0 = np.zeros((Nmax+1, Nmax+1))
    for i in range(L2_L0.shape[0]):
        q_L2_L0[L2_L0[i,0].astype(int), L2_L0[i,2].astype(int)] = L2_L0[i,8]
        q_L2_L0[L0_L2[i,2].astype(int), L0_L2[i,0].astype(int)] = L0_L2[i,8]
        
    # Quadratische L=2, L'=2 Matrix
    q_L2_L2 = np.zeros((Nmax+1, Nmax+1))
    for i in range(L2_L2.shape[0]):
        q_L2_L2[L2_L2[i,2].astype(int), L2_L2[i,0].astype(int)] = L2_L2[i,8]
        q_L2_L2[L2_L2[i,0].astype(int), L2_L2[i,2].astype(int)] = L2_L2[i,8]
    
    # Zusammenfassen der einzelenen L, L' Mitrizen zu einer
    deut_mat = np.vstack((np.hstack((q_L0_L0,q_L0_L2)),np.hstack((q_L2_L0,q_L2_L2))))
    
    return deut_mat