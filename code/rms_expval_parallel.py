import numpy as np
import matplotlib.pyplot as plt
from make_matrix import mat
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from math import sqrt


chi_Nmax150_aHO12 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.20.meho', skiprows=1)
chi_Nmax150_aHO14 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.40.meho', skiprows=1)
chi_Nmax150_aHO16 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
chi_Nmax150_aHO18 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO01.80.meho', skiprows=1)
chi_Nmax150_aHO20 = np.loadtxt(r'matrix_elements/chi2bM3A_srg0000_NMax150_LMax02_aHO02.00.meho', skiprows=1)
trel_Nmax150 = np.loadtxt(r'matrix_elements/trel_NMax150_LMax02_aHO01.00.meho', skiprows=1)
rsq_Nmax150 = np.loadtxt(r'matrix_elements/rsq_NMax150_LMax02.meho', skiprows=1)

chi = np.array([chi_Nmax150_aHO12, chi_Nmax150_aHO14, chi_Nmax150_aHO16, chi_Nmax150_aHO18, chi_Nmax150_aHO20])
aHO = np.array([1.2,1.4,1.6,1.8,2])


def rms_Nmax(i, j, rsq_Nmax150, aHO_sq, aHO_inv_sq):
    rsq = 0
    h_eigvals, h_eigvecs = np.linalg.eig(mat(chi[j], i) + mat(trel_Nmax150, i) * aHO_inv_sq)
    gs_index = np.where(h_eigvals == min(h_eigvals))[0][0]
    
    mat_rsq = mat(rsq_Nmax150, i) * aHO_sq/4
    
    rsq = np.dot(h_eigvecs[:, gs_index], np.dot(mat_rsq, h_eigvecs[:, gs_index]))
    rms = np.sqrt(rsq)
    
    return [i, rms]

def rms_aHO(j, Nstep):
    rms = []
    aHO_sq = aHO[j]**2
    aHO_inv_sq = 1 / aHO_sq
    
    Nmax_lst = np.arange(0, 151, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rms_Nmax, i, j, rsq_Nmax150, aHO_sq, aHO_inv_sq) for i in Nmax_lst]
        for future in futures:
            rms.append(future.result())
    
    return rms

rms = np.array(Parallel(n_jobs=-1)(delayed(rms_aHO)(j, 1) for j in range(aHO.shape[0])))
Rrms = np.mean(rms[:, 150, 1])
print(Rrms)

plt.figure()

# aHO = 1.2 fm
plt.subplot(231)
plt.plot(rms[0, :, 0], rms[0,::, 1], 'x-')
plt.title('$R_{rms}$ bei $a_{HO}$=1.2 fm')
plt.xlabel('$N_{max}$')
plt.xlim(0, 60)
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, rms[0, 150, 1], 2.0, 2.2])
plt.grid(linestyle='--')

# aHO = 1.4 fm
plt.subplot(232)
plt.plot(rms[1, :, 0], rms[1,::, 1], 'x-')
plt.title('$R_{rms}$ bei $a_{HO}$=1.4 fm')
plt.xlabel('$N_{max}$')
plt.xlim(0, 40)
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, rms[1, 150, 1], 2.0, 2.2])
plt.grid(linestyle='--')

# aHO = 1.6 fm
plt.subplot(233)
plt.plot(rms[2, :, 0], rms[2,::, 1], 'x-')
plt.title('$R_{rms}$ bei $a_{HO}$=1.6 fm')
plt.xlabel('$N_{max}$')
plt.xlim(0, 30)
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, rms[2, 150, 1], 2.0, 2.2])
plt.grid(linestyle='--')

# aHO = 1.8 fm
plt.subplot(234)
plt.plot(rms[3, :, 0], rms[3,::, 1], 'x-')
plt.title('$R_{rms}$ bei $a_{HO}$=1.8 fm')
plt.xlabel('$N_{max}$')
plt.xlim(0, 25)
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, rms[3, 150, 1], 2.0, 2.2])
plt.grid(linestyle='--')

# aHO = 2.0 fm
plt.subplot(235)
plt.plot(rms[4, :, 0], rms[4,::, 1], 'x-')
plt.title('$R_{rms}$ bei $a_{HO}$=2.0 fm')
plt.xlabel('$N_{max}$')
plt.xlim(0, 30)
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, rms[4, 150, 1], 2.0, 2.2])
plt.grid(linestyle='--')

plt.subplot(236)
for i in range(rms.shape[0]):
    plt.plot(rms[i,::, 0],rms[i,::, 1], 'x-')
legend = np.array([])
for i in aHO:
    legend = np.append(legend,'$a_{HO}$ = '+ str(i) + ' fm')
plt.title('$R_{rms}$ bei verschiedenen $a_{HO}$')
plt.legend(legend)
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, Rrms, 2.0, 2.2])
plt.xlabel('$N_{max}$')
plt.xlim(0, 60)
plt.grid(linestyle='--')


plt.figure()

# Nmax = 0
plt.subplot(231)
plt.plot(aHO, rms[:, 0, 1], 'x-')
plt.title('$R_{rms}$ bei $N_{max}$=0')
plt.xlabel('$a_{HO} (fm)$')
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, Rrms, 2.0, 2.2])
plt.grid(linestyle='--')

# Nmax = 30
plt.subplot(232)
plt.plot(aHO, rms[:, 30, 1], 'x-')
plt.title('$R_{rms}$ bei $N_{max}$=30')
plt.xlabel('$a_{HO} (fm)$')
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, Rrms, 2.0, 2.2])
plt.grid(linestyle='--')

# Nmax = 60
plt.subplot(233)
plt.plot(aHO, rms[:, 60, 1], 'x-')
plt.title('$R_{rms}$ bei $N_{max}$=60')
plt.xlabel('$a_{HO} (fm)$')
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, Rrms, 2.0, 2.2])
plt.grid(linestyle='--')

# Nmax = 90
plt.subplot(234)
plt.plot(aHO, rms[:, 90, 1], 'x-')
plt.title('$R_{rms}$ bei $N_{max}$=90')
plt.xlabel('$a_{HO} (fm)$')
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, Rrms, 2.0, 2.2])
plt.grid(linestyle='--')

# Nmax = 120
plt.subplot(235)
plt.plot(aHO, rms[:, 120, 1], 'x-')
plt.title('$R_{rms}$ bei $N_{max}$=120')
plt.xlabel('$a_{HO} (fm)$')
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, Rrms, 2.0, 2.2])
plt.grid(linestyle='--')

Nmax_step30 = np.arange(0, 121, 30)
plt.subplot(236)
for i in Nmax_step30:
    plt.plot(aHO, rms[:, i, 1], 'x-')
legend = np.array([])
for i in Nmax_step30:
    legend = np.append(legend,'$N_{max}$ = '+ str(i))
plt.title('$R_{rms}$ bei verschiedenen $N_{max}$')
plt.legend(legend)
plt.ylabel("$R_{rms}$ (fm)")
plt.yticks([1.2, 1.4, 1.6, 1.8, Rrms, 2.0, 2.2])
plt.xlabel('$a_{HO}$ (fm)')
plt.grid(linestyle='--')

plt.show()
