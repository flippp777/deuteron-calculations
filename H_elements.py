import numpy as np
from matplotlib import pyplot as plt
from sys import platform

if platform == "win32":
    chi_Nmax20 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax20_LMax02_hwHO020.meho', skiprows=1)
    trel_Nmax20 = np.loadtxt(r'Matrixelemente\trel_NMax20_LMax02_hwHO020.meho', skiprows=1)

    chi_Nmax150_aHO12 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
    chi_Nmax150_aHO14 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
    chi_Nmax150_aHO16 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
    chi_Nmax150_aHO18 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
    chi_Nmax150_aHO20 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_hwHO020.meho', skiprows=1)

    trel_Nmax150 = np.loadtxt(r'Matrixelemente\trel_NMax150_LMax02_hwHO020.meho', skiprows=1)
else:
    chi_Nmax20 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax20_LMax02_hwHO020.meho', skiprows=1)
    trel_Nmax20 = np.loadtxt(r'Matrixelemente/trel_NMax20_LMax02_hwHO020.meho', skiprows=1)

    chi_Nmax150_aHO12 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
    chi_Nmax150_aHO14 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
    chi_Nmax150_aHO16 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
    chi_Nmax150_aHO18 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
    chi_Nmax150_aHO20 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_hwHO020.meho', skiprows=1)

    

# Erschtellt die Matirx mit gegebenen Matrixelementen
def mat(mat_el, Nmax):
    #Conditions für den Deuterium GZ und N, N' truncation
    con_SJTMT = (
        (mat_el[:,0] <= Nmax) & # N <= Nmax
        (mat_el[:,2] <= Nmax) & # N' <= Nmax
        (mat_el[:, 4] == 1) &   # S=1
        (mat_el[:, 5] == 1) &   # J=1
        (mat_el[:, 6] == 0) &   # T=0
        (mat_el[:, 7] == 0))    # MT=0
    
    c_mat_el1 = mat_el[con_SJTMT]
    
    print(np.transpose(np.array([c_mat_el1[:,0],c_mat_el1[:,1],c_mat_el1[:,2],c_mat_el1[:,3],c_mat_el1[:,8]])))
    
    return np.transpose(np.array([c_mat_el1[:,0],c_mat_el1[:,1],c_mat_el1[:,2],c_mat_el1[:,3],c_mat_el1[:,8]]))



np.savetxt('chi_elements_test.txt', mat(chi_Nmax150_aHO14, 1))


# H_Nmax150_aHO14 = mat(chi_Nmax150_aHO14, 100) + mat(trel_Nmax150, 100)*1/1.4**2
# eval_Nmax150_aHO14, evec_Nmax150_aHO14 = np.linalg.eig(H_Nmax150_aHO14)

# print(min(eval_Nmax150_aHO14))