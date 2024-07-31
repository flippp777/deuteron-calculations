import numpy as np
import matplotlib.pyplot as plt
from create_matrix import mat

rsq_Nmax150 = np.loadtxt(r'Matrixelemente/rsq_NMax150_LMax02.meho', skiprows=1)
chi_Nmax150_aHO16 = np.loadtxt(r'Matrixelemente\chi2bM3A_srg0000_NMax150_LMax02_aHO01.60.meho', skiprows=1)
trel_Nmax150 = np.loadtxt(r'Matrixelemente\trel_NMax150_LMax02.meho', skiprows=1)

chi_mat = mat(chi_Nmax150_aHO16, 60)
trel_mat = mat(trel_Nmax150, 60) * 1/1.6**2
rsq_mat = mat(rsq_Nmax150, 60) * 1.6**2

aHO = np.array([1.2,1.4,1.6,1.8,2])

rsq_eigval, rsq_eigvec = np.linalg.eig(rsq_mat)

def plot_rms():
    Nmax_step = np.arange(2,151,2)
    rsq = np.zeros((2,len(Nmax_step)))

    for j in aHO:
        rsqm = np.array([])
        for i in Nmax_step:
            rsqm = np.append(rsqm, [np.sqrt(min(np.linalg.eig(mat(rsq_Nmax150, i) * j**2)[0]))], axis=0)
        rsq = np.append(rsq, [rsqm], axis=0)
    rsq=np.delete(rsq, [0,1], axis=0)

    for i in range(rsq.shape[0]):
        plt.scatter(Nmax_step,rsq[i])

    legend = np.array([])
    for i in aHO:
        legend = np.append(legend,'$a_{HO}$ = '+ str(i) + ' fm')

    plt.title('$R_{rms}$ bei verschiedenen $a_{HO}$')
    plt.legend(legend)
    plt.ylabel("$R_{rms}$ fm")
    plt.xlabel('$N_{max}$')

# plot_rms()

plt.show()