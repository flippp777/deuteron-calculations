import numpy as np
import matplotlib.pyplot as plt
from create_matrix import mat

chi_Nmax150_aHO20 = np.loadtxt(r'Matrixelemente/chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
trel_Nmax150 = np.loadtxt(r'Matrixelemente/trel_NMax150_LMax02_aHO01.00.meho', skiprows=1)
rsq_Nmax150 = np.loadtxt(r'Matrixelemente/rsq_NMax150_LMax02.meho', skiprows=1)

rsq_elements_L0 = rsq_Nmax150[((rsq_Nmax150[:, 1] == 0) &   # L=0
                                (rsq_Nmax150[:, 3] == 0) &  # L'=0
                                (rsq_Nmax150[:, 4] == 1) &  # S=1
                                (rsq_Nmax150[:, 5] == 1) &  # J=1
                                (rsq_Nmax150[:, 6] == 0) &  # T=0
                                (rsq_Nmax150[:, 7] == 0) &  # MT=0
                                (rsq_Nmax150[:,8] != 0))]   # rsq !=0
np.savetxt('rsq_Nmax140_L0.txt', rsq_elements_L0)
rsq_elements_L2 = rsq_Nmax150[((rsq_Nmax150[:, 1] == 2) &  # L=2
                                (rsq_Nmax150[:, 3] == 2) &  # L'=2
                                (rsq_Nmax150[:, 4] == 1) &  # S=1
                                (rsq_Nmax150[:, 5] == 1) &  # J=1
                                (rsq_Nmax150[:, 7] == 0) &  # MT=0
                                (rsq_Nmax150[:,8] != 0))]   # rsq !=0
np.savetxt('rsq_Nmax140_L2.txt', rsq_elements_L2)

aHO = 2

Nmax_lst = np.arange(5,41,5)

rms2 = []
for i in Nmax_lst:
    chi_mat = mat(chi_Nmax150_aHO20, i)
    trel_mat = mat(trel_Nmax150, i) * 1/(aHO**2)

    h_eigvals, h_eigvecs = np.linalg.eig(chi_mat + trel_mat)
    gs_index = np.where(h_eigvals == min(h_eigvals))[0][0]
    gs_vec = h_eigvecs[gs_index]
    
    rsq1 = 0
    for N1 in range(i):
        for N2 in range(i):
            rsqN1N2_elements1 = rsq_elements_L0[((rsq_elements_L0[:, 0] == N1) & (rsq_elements_L0[:, 2] == N2))]  
            if rsqN1N2_elements1.shape != (0,9) and N1 == N2:
                rsq1 += rsqN1N2_elements1[0,8] * gs_vec[N1] * gs_vec[N2] * aHO**2/4
            elif rsqN1N2_elements1.shape != (0,9):
                rsq1 += 2 * rsqN1N2_elements1[0,8] * gs_vec[N1] * gs_vec[N2] * aHO**2/4
                
    rsq2 = 0
    for N1 in range(i):
        for N2 in range(i):
            rsqN1N2_elements2 = rsq_elements_L2[((rsq_elements_L2[:, 0] == N1) & (rsq_elements_L2[:, 2] == N2))]  
            if  rsqN1N2_elements2.shape != (0,9) and N1 == N2:
                rsq2 += rsqN1N2_elements2[0,8] * gs_vec[i + 1 + N1] * gs_vec[i + 1 + N2]  * aHO**2/4
            elif rsqN1N2_elements2.shape != (0,9):
                rsq1 += 2 * rsqN1N2_elements2[0,8] * gs_vec[i + 1 + N1] * gs_vec[i + 1 + N2] * aHO**2/4
    
    rms2.append([i,np.sqrt(rsq1 + rsq2)])

rms2 = np.array(rms2)

plt.figure()
plt.subplot(221)
plt.plot(rms2[:, 0],rms2[:, 1])
plt.title('$R_{rms}$ bei $a_{HO}$ = 2 fm, elemetwise')
plt.ylabel("$R_{rms}$ in fm")
plt.xlabel('$N_{max}$')


def rms_Nmax(Nmax, chi, rsq_Nmax150, aHO_sq, aHO_inv_sq):
    rsq = 0
    h_eigvals, h_eigvecs = np.linalg.eig(mat(chi, Nmax) + mat(trel_Nmax150, Nmax) * aHO_inv_sq)
    gz_index = np.where(h_eigvals == min(h_eigvals))[0][0]
    
    mat_rsq = mat(rsq_Nmax150, Nmax) * aHO_sq/4
    
    rsq = np.dot(h_eigvecs[gz_index], np.dot(mat_rsq, h_eigvecs[gz_index]))
    rms = np.sqrt(rsq)
    
    return [Nmax, rms]

def rms_aHO(Nstep):
    rms = []
    aHO_sq = aHO**2
    aHO_inv_sq = 1 / aHO_sq
    
    Nmax_lst = np.arange(Nstep, 41, Nstep)
    
    for i in Nmax_lst:
            rms.append(rms_Nmax(i, chi_Nmax150_aHO20, rsq_Nmax150, aHO_sq, aHO_inv_sq))
    
    return rms

rms = np.array(rms_aHO(5))

plt.subplot(222)
plt.plot(rms[:, 0],rms[:, 1])
plt.title('$R_{rms}$ bei $a_{HO}$ = 2 fm, matrix')
plt.ylabel("$R_{rms}$ in fm")
plt.xlabel('$N_{max}$')

plt.subplot(223)
plt.plot(rms[:, 0],rms[:, 1])
plt.plot(rms2[:, 0],rms2[:, 1])
plt.title('$R_{rms}$ bei $a_{HO}$ = 2 fm')
plt.legend(["Matrix", "Elementwise"])
plt.ylabel("$R_{rms}$ in fm")
plt.xlabel('$N_{max}$')

plt.show()