import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from sys import platform

if platform == "win32":
    chi_Nmax150_aHO12 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
    chi_Nmax150_aHO14 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
    chi_Nmax150_aHO16 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
    chi_Nmax150_aHO18 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
    chi_Nmax150_aHO20 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
    trel_Nmax150 = np.loadtxt(r'Matrixelemente\trel_NMax150_LMax02_aHO01.00.meho', skiprows=1)

else: 
    chi_Nmax150_aHO12 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
    chi_Nmax150_aHO14 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
    chi_Nmax150_aHO16 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
    chi_Nmax150_aHO18 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
    chi_Nmax150_aHO20 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
    trel_Nmax150 = np.loadtxt(r'Matrixelemente/trel_NMax150_LMax02_aHO01.00.meho', skiprows=1)


chi = np.array([chi_Nmax150_aHO12, chi_Nmax150_aHO14, chi_Nmax150_aHO16, chi_Nmax150_aHO18, chi_Nmax150_aHO20])
aHO = np.array([1.2,1.4,1.6,1.8,2])


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

# Gibt den niedrigsten Eigenwert aus
def E0_Nmax(i, j):
    h = mat(chi[j], i) + mat(trel_Nmax150, i)*1/(aHO[j]**2)
    E0 = min(np.linalg.eigvalsh(h))
    return E0

testmat = mat(chi_Nmax150_aHO20, 150)
eigvec, eigval = np.linalg.eig(testmat + mat(trel_Nmax150,150)*1/2**2)

Nmax_step75 = np.array([75,150])
Nmax_step30 = np.arange(30, 180, 30)
Nmax_step2 = np.arange(2,152,2)

def E0_aHO(j):
    E0 = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(E0_Nmax, j, i) for i in Nmax_step2]
        for future in futures:
            E0.append(future.result())
    return E0
    


def plotsE0Nmax():
    # E0_Nmax = np.zeros((2,75))
    # for m in range(chi.shape[0]):
    #     E0_Nmaxm = np.array([])
    #     for i in Nmax_step2:
    #         E0_Nmaxm = np.append(E0_Nmaxm, [E0(chi[m], aHO[m], i)], axis=0)
    #     E0_Nmax = np.append(E0_Nmax, [E0_Nmaxm], axis = 0)
    # E0_Nmax = np.delete(E0_Nmax,[0,1], axis=0)
    
    E0 = np.array(Parallel(n_jobs=-1)(delayed(E0_aHO)(j) for j in range(aHO.shape[0])))

    # Plot
    plt.figure()

    # aHO=1.2
    plt.subplot(231)
    plt.scatter(Nmax_step2,E0[0])
    plt.title('$E_0$ bei $a_{HO}$=1.2 fm')
    plt.xlabel('$N_{max}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

    # aHO = 1.4
    plt.subplot(232)
    plt.scatter(Nmax_step2,E0[1])
    plt.title('$E_0$ bei $a_{HO}$=1.4 fm')
    plt.xlabel('$N_{max}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

    # aHO = 1.6
    plt.subplot(233)
    plt.scatter(Nmax_step2,E0[2])
    plt.title('$E_0$ bei $a_{HO}$=1.6 fm')
    plt.xlabel('$N_{max}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

    # aHO = 1.8
    plt.subplot(234)
    plt.scatter(Nmax_step2,E0[3])
    plt.title('$E_0$ bei $a_{HO}$=1.8 fm')
    plt.xlabel('$N_{max}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

    # aHO = 2.0
    plt.subplot(235)
    plt.scatter(Nmax_step2,E0[4])
    plt.title('$E_0$ bei $a_{HO}$=2.0 fm')
    plt.xlabel('$N_{max}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')
    
    # alle zusammen
    plt.subplot(236)
    for i in range(E0.shape[0]):
        plt.scatter(Nmax_step2,E0[i])

    legend = np.array([])
    for i in aHO:
        legend = np.append(legend,'$a_{HO}$ = '+ str(i) + ' fm')

    plt.title('$E_0$ bei verschiedenen $a_{HO}$')
    plt.legend(legend)
    plt.xlabel('$N_{max}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

 
def plotsE0aHO():
    E0_aHO = np.zeros((2,5))
    for m in Nmax_step30:
        E0_aHOm = np.array([])
        for i in range(chi.shape[0]):
            E0_aHOm = np.append(E0_aHOm, [E0(chi[i], aHO[i], m)], axis=0)
        E0_aHO = np.append(E0_aHO, [E0_aHOm], axis = 0)

    E0_aHO = np.delete(E0_aHO,[0,1], axis=0)
        # Plot
    plt.figure()

    # aHO=1.2
    plt.subplot(231)
    plt.scatter(aHO,E0_aHO[0])
    plt.title('$E_0$ bei $N_{max}$=30')
    plt.xlabel('$a_{HO}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

    # aHO = 1.4
    plt.subplot(232)
    plt.scatter(aHO,E0_aHO[1])
    plt.title('$E_0$ bei $N_{max}$=60')
    plt.xlabel('$a_{HO}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

    # aHO = 1.6
    plt.subplot(233)
    plt.scatter(aHO,E0_aHO[2])
    plt.title('$E_0$ bei $N_{max}$=90')
    plt.xlabel('$a_{HO}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

    # aHO = 1.8
    plt.subplot(234)
    plt.scatter(aHO,E0_aHO[3])
    plt.title('$E_0$ bei $N_{max}$=120')
    plt.xlabel('$a_{HO}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

    # aHO = 2.0
    plt.subplot(235)
    plt.scatter(aHO,E0_aHO[4])
    plt.title('$E_0$ bei $N_{max}$=150')
    plt.xlabel('$a_{HO}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')
    
    # alle zusammen
    plt.subplot(236)
    for i in range(E0_aHO.shape[0]):
        plt.scatter(aHO,E0_aHO[i])

    legend = np.array([])
    for i in Nmax_step30:
        legend = np.append(legend,'$N_{max}$ = '+ str(i))

    plt.title('$E_0$ bei verschiedenen $N_{max}$')
    plt.legend(legend)
    plt.xlabel('$a_{HO}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

plotsE0Nmax()
# plotsE0aHO()

plt.show()