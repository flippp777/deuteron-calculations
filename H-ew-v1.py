import numpy as np
from matplotlib import pyplot as plt

chi_Nmax20 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax20_LMax02_hwHO020.meho', skiprows=1)
trel_Nmax20 = np.loadtxt(r'Matrixelemente\trel_NMax20_LMax02_hwHO020.meho', skiprows=1)

chi_Nmax150_aHO12 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
chi_Nmax150_aHO14 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
chi_Nmax150_aHO16 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
chi_Nmax150_aHO18 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
chi_Nmax150_aHO20 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_hwHO020.meho', skiprows=1)

trel_Nmax150 = np.loadtxt(r'Matrixelemente\trel_NMax150_LMax02.meho', skiprows=1)

# Erstellt quadratische Matrix und ersetzt die Elemente anhand der Indizes N und N'
def quad_mat(mat_el, Nmax):
    
    q_mat = np.zeros((Nmax+1, Nmax+1))
    
    for i in range(mat_el.shape[0]):
        q_mat[mat_el[i,0].astype(int), mat_el[i,2].astype(int)] = mat_el[i,8]
    
    return q_mat

# Erschtellt die Matirx mit gegebenen Matrixelementen
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
    
    # Quadratische Matrizen mit den verschiedenen L, L' kombinationen
    q_L0_L0 = quad_mat(L0_L0, Nmax)
    q_L0_L2 = quad_mat(L0_L2, Nmax)
    q_L2_L0 = quad_mat(L2_L0, Nmax)
    q_L2_L2 = quad_mat(L2_L2, Nmax)
    
    # Zusammenfassen der einzelenen L, L' Mitrizen zu einer
    deut_mat = np.vstack((np.hstack((q_L0_L0,q_L0_L2)),np.hstack((q_L2_L0,q_L2_L2))))
    
    return deut_mat


H_Nmax150_aHO14 = mat(chi_Nmax150_aHO14, 100)  + mat(trel_Nmax150, 100)*1/(1.4**2)
eval_Nmax150_aHO14, evec_Nmax150_aHO14 = np.linalg.eig(H_Nmax150_aHO14)
print(min(eval_Nmax150_aHO14))

H_Nmax150_aHO20 = mat(chi_Nmax150_aHO20, 150) + mat(trel_Nmax150, 150)*1/2**2
eval_Nmax150_aHO20, evec_Nmax150_aHO20 = np.linalg.eig(H_Nmax150_aHO20)
print(min(eval_Nmax150_aHO20))

Nmax_lst = np.arange(15, 165, 15)
GZ_E = np.array([])

for i in Nmax_lst:
    eigval = min(np.linalg.eig(mat(chi_Nmax150_aHO14, i) + mat(trel_Nmax150, i)*1/1.4**2)[0])
    GZ_E = np.append(GZ_E,eigval)


Nmax_E_lst = np.array([[]])
print(GZ_E)
for i in range(GZ_E.shape[0]):
    Nmax_E_lst = np.append(Nmax_E_lst, [Nmax_lst[i], GZ_E[i]])
    
print(Nmax_E_lst)