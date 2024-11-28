import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sys import exit


chi_Nmax150_aHO12 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
chi_Nmax150_aHO14 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
chi_Nmax150_aHO16 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
chi_Nmax150_aHO18 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
chi_Nmax150_aHO20 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
chi_Nmax150_aHO24 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO02.40.meho', skiprows=1)
chi_Nmax150_aHO26 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO02.60.meho', skiprows=1)
trel_Nmax150 = np.loadtxt(r'matrix_elements/trel_NMax150_LMax02_aHO01.00.meho', skiprows=1)


chi = np.array([chi_Nmax150_aHO12, chi_Nmax150_aHO14, chi_Nmax150_aHO16, chi_Nmax150_aHO18, chi_Nmax150_aHO20, chi_Nmax150_aHO24, chi_Nmax150_aHO26])
aHO = np.array([1.2,1.4,1.6,1.8,2.0,2.4,2.6])


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
def E0(chi, aHO, Nmax):
    if Nmax > 150:
        exit('Nmax ist maximal 150')
    elif aHO not in [1.2,1.4,1.6,1.8,2.0,2.4,2.6]:
        exit('Es gibt nur aHO 1.2, 1.4, 1.6, 1.8, 2')
    else:
        E0 = min(np.linalg.eigvalsh(mat(chi, Nmax) + mat(trel_Nmax150, Nmax)*1/(aHO**2)))
        return E0
    
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


testmat = mat(chi_Nmax150_aHO20, 150)
eigvec, eigval = np.linalg.eig(testmat + mat(trel_Nmax150,150)*1/2**2)
    

Nmax_step75 = np.array([75,150])
Nmax_step30 = np.arange(0, 121, 30)
Nmax_step2 = np.arange(0,152,2)

def plotsE0Nmax(Nmax_lst, aHO_lst, save=False):
    E0_Nmax = []
    for m in range(len(aHO_lst)):
        E0_Nmaxm = []
        for Nmax in Nmax_lst:
            E0_Nmaxm.append(E0(chi[m], aHO_lst[m], Nmax))
        E0_Nmax.append(E0_Nmaxm)

    E0_Nmax = np.array(E0_Nmax) 

    red = '#FF0000'
    blue = '#0000FF'
    black = '#000000'
    purple = '#FF00FF'
    n = len(aHO_lst)
    
    plt.figure(figsize=(12,7.45))
    for i in range(E0_Nmax.shape[0]):
        plt.plot(Nmax_lst,E0_Nmax[i], 'x-', color=colorFader(purple, black, i/n))

    legend = np.array([])
    for i in aHO:
        legend = np.append(legend,'$a_{HO}$ = '+ str(i) + ' fm')

    # plt.title('$E_0$ bei verschiedenen $a_{HO}$')
    plt.legend(legend)
    plt.xlabel('$N_{max}$')
    plt.ylabel('$E_0$ (MeV)')
    plt.yticks([-2.225, -2, -1, 0, 1, 2, 3])
    plt.grid(linestyle='--')
    # plt.xlim(0,25)
    # plt.ylim(-2.5, 3.5)

    if save: 
        plt.savefig(f'E0_Nmax.pdf')
        

 
def plotsE0aHO(Nmax_lst, aHO_lst, save=False):
    E0_aHO = []
    for m in Nmax_lst:
        E0_aHOm = []
        for i in range(len(aHO_lst)):
            E0_aHOm.append(E0(chi[i], aHO[i], m))
        E0_aHO.append(E0_aHOm)

    E0_aHO = np.array(E0_aHO)

    red = '#FF0000'
    blue = '#0000FF'
    black = '#000000'
    purple = '#FF00FF'
    n = len(Nmax_lst)
    
    plt.figure(figsize=(12,7.45))
    for i in range(E0_aHO.shape[0]):
        plt.plot(aHO_lst, E0_aHO[i], 'x-', color=colorFader(purple, black, i/n))

    legend = np.array([])
    for i in Nmax_lst:
        legend = np.append(legend,'$N_{max}$ = '+ str(i))

    plt.legend(legend)
    plt.yticks([-2.225, -2, -1, 0, 1, 2, 3])
    plt.xticks([1.2, 1.4, 1.6, 1.8, 2])
    plt.xlabel('$a_{HO}$ (fm)')
    plt.ylabel('$E_0$ (MeV)')
    plt.grid(linestyle='--')

    if save: 
        plt.savefig(f'E0_aHO.pdf')


plotsE0Nmax(np.arange(0, 26,1), [1.2, 1.4, 1.6, 1.8, 2], True)

plotsE0aHO(np.arange(0, 11, 1), [1.2, 1.4, 1.6, 1.8, 2], True)

plt.show()