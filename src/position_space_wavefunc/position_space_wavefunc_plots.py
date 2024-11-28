import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import sys

if sys.platform == "win32":
    sys.path.insert(1, r'C:\Users\phili.DESKTOP\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc


# print('density integral aHO = 1.2')
# for i in range(0,40,10):
#     print(integrate.quad(lambda r: deuteron_wavefunc(i, 1.2, r)[3] * r**2, 0, np.inf))
# print('density integral aHO = 1.4')
# for i in range(0,40,10):
#     print(integrate.quad(lambda r: deuteron_wavefunc(i, 1.4, r)[3] * r**2, 0, np.inf))
# print('density integral aHO = 1.6')
# for i in range(0,40,10):
#     print(integrate.quad(lambda r: deuteron_wavefunc(i, 1.6, r)[3] * r**2, 0, np.inf))
# print('density integral aHO = 1.8')
# for i in range(0,40,10):
#     print(integrate.quad(lambda r: deuteron_wavefunc(i, 1.8, r)[3] * r**2, 0, np.inf))
# print('density integral aHO = 2.0')
# for i in range(0,40,10):
#     print(integrate.quad(lambda r: deuteron_wavefunc(i, 2.0, r)[3] * r**2, 0, np.inf))

def derivative(x, f, n):
    return np.gradient(f, x=x)


def plot_wavefunc_Nmax(Nmax, aHO):
    x_model = np.linspace(0, 15, 500)
    
    plt.figure()
    
    # Nmax = 0
    wavefuncs_Nmax = deuteron_wavefunc(Nmax, aHO, x_model)
    wavefunc_Nmax, wavefunc_Nmax_L0, wavefunc_Nmax_L2 = wavefuncs_Nmax[0], wavefuncs_Nmax[1], wavefuncs_Nmax[2]

    plt.plot(x_model, wavefunc_Nmax)
    plt.plot(x_model, wavefunc_Nmax_L0)
    plt.plot(x_model, wavefunc_Nmax_L2)
    plt.title('Wellenfunctionen bei $a_{HO} = $' + str(aHO) +' fm für $N_{max} = $' + str(Nmax))
    plt.legend(['$\Phi(r)$', '$\Phi_{L=0}(r)$', '$\Phi_{L=2}(r)$'])
    plt.xlabel('r (fm)')
    plt.grid(linestyle='--')

def plots_wavefunc_Nmax(aHO, NmaxMax, step):
    x_model = np.linspace(0, 15, 500)
    
    plt.figure()
    
    for N in range(0, NmaxMax + 1, step):
        wavefunc_L0 = np.abs(deuteron_wavefunc(N, aHO, x_model)[1])
        plt.plot(x_model, wavefunc_L0) 
    for N in range(0, NmaxMax + 1, step):
        wavefunc_L2 = np.abs(deuteron_wavefunc(N, aHO, x_model)[2])
        plt.plot(x_model, wavefunc_L2) 
    plt.xlim(right=10)
    plt.title('Wellenfunktionen bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$')
    plt.xlabel('r (fm)')
    plt.legend(np.array([['$\Phi_{L=0}(r)$, $N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)], ['$\Phi_{L=2}(r)$, $N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)]]).flatten())
    plt.grid(linestyle='--')
    
def plots_density_Nmax(aHO, NmaxMax, step):
    x_model = np.linspace(0, 15, 500)
    
    plt.figure()
    
    for N in range(0, NmaxMax + 1, step):
        plt.plot(x_model, deuteron_wavefunc(N, aHO, x_model)[3])
    plt.xlim(0, 8)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$')
    plt.xlabel('r (fm)')
    plt.ylabel('$|\Phi(r)|^2$')
    plt.legend(['$N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)])
    plt.grid(linestyle='--')

def logplots_density_Nmax(aHO, NmaxMax, step, xlim):
    x_model_log = np.linspace(0, 20, 500)
    
    plt.figure()
    
    for N in range(0, NmaxMax + 1, step):
        plt.semilogy(x_model_log, deuteron_wavefunc(N, aHO, x_model_log)[3])
    plt.ylim(bottom=10**(-6))
    plt.xlim(0, xlim)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$ mit logarithmischer y-Achse')
    plt.xlabel('r (fm)')
    plt.ylabel('$|\Phi(r)|^2$')
    plt.legend(['$N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)])
    plt.grid(linestyle='--')

def logplots_ddensity_dr_Nmax(aHO, NmaxMax, step, xlim):
    x_model_log = np.linspace(0, xlim, 1000)
   
    plt.figure()
    
    for N in range(0, NmaxMax + 1, step):
        plt.plot(x_model_log, derivative(x_model_log, np.log(deuteron_wavefunc(N, aHO, x_model_log)[3]), 1))
    # plt.ylim(bottom=10**(-6))
    plt.xlim(0, xlim)
    plt.title('Ableitung der logarithmischen Aufenthaltswahrscheinlichkeit nach r bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$')
    plt.xlabel('r (fm)')
    plt.ylabel('d/dr ln$(|\Phi(r)|^2)$')
    plt.legend(['$N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)])
    plt.grid(linestyle='--')

aHO = [1.2, 1.4, 1.6, 1.8, 2]

def plot_wavefunc_aHO_L0(Nmax, aHO):
    x_model = np.linspace(0, 15, 500)
    
    plt.plot(x_model, np.abs(deuteron_wavefunc(Nmax, aHO, x_model)[1]))
    plt.title('Wellenfunktionsanteile für $N_{max} = $' + str(Nmax))
    plt.xlabel('r (fm)')

def plot_wavefunc_aHO_L2(Nmax, aHO):
    x_model = np.linspace(0, 15, 500)
    
    plt.plot(x_model, np.abs(deuteron_wavefunc(Nmax, aHO, x_model)[2]))
    plt.title('Wellenfunktionsanteile für $N_{max} = $' + str(Nmax))
    plt.xlabel('r (fm)')

def plots_wavefunc_aHO():
    plt.figure()
    
    plt.subplot(221)
    for i in aHO:
        plot_wavefunc_aHO_L0(0, i)
    for i in aHO:
        plot_wavefunc_aHO_L2(0, i)
    plt.legend(np.array([['$\Phi_{L=0}(r), a_{HO} = $' + str(i) + ' fm' for i in aHO], ['$\Phi_{L=2}(r), a_{HO} = $' +str(i) + ' fm' for i in aHO]]).flatten())
    
    plt.subplot(222)
    for i in aHO:
        plot_wavefunc_aHO_L0(10, i)
    for i in aHO:
        plot_wavefunc_aHO_L2(10, i)
    plt.legend(np.array([['$\Phi_{L=0}(r), a_{HO} = $' + str(i) + ' fm' for i in aHO], ['$\Phi_{L=2}(r), a_{HO} = $' +str(i) + ' fm' for i in aHO]]).flatten())

    plt.subplot(223)
    for i in aHO:
        plot_wavefunc_aHO_L0(20, i)
    for i in aHO:
        plot_wavefunc_aHO_L2(20, i)
    plt.legend(np.array([['$\Phi_{L=0}(r), a_{HO} = $' + str(i) + ' fm' for i in aHO], ['$\Phi_{L=2}(r), a_{HO} = $' +str(i) + ' fm' for i in aHO]]).flatten())

    plt.subplot(224)
    for i in aHO:
        plot_wavefunc_aHO_L0(30, i)
    for i in aHO:
        plot_wavefunc_aHO_L2(30, i)
    plt.legend(np.array([['$\Phi_{L=0}(r), a_{HO} = $' + str(i) + ' fm' for i in aHO], ['$\Phi_{L=2}(r), a_{HO} = $' +str(i) + ' fm' for i in aHO]]).flatten())
    

def plot_denstiy_aHO(Nmax, aHO):
    x_model = np.linspace(0, 15, 500)
    
    plt.plot(x_model, np.abs(deuteron_wavefunc(Nmax, aHO, x_model)[3]))
    plt.title('Aufenthaltswahrscheinlichkeit für $N_{max} = $' + str(Nmax))
    plt.xlabel('r (fm)')

def plots_density_aHO():
    plt.figure()
    
    plt.subplot(221)
    for i in aHO:
        plot_denstiy_aHO(0, i)
    plt.legend(['$a_{HO} = $' + str(i) + ' fm' for i in aHO])
    
    plt.subplot(222)
    for i in aHO:
        plot_denstiy_aHO(10, i)
    plt.legend(['$a_{HO} = $' + str(i) + ' fm' for i in aHO])

    plt.subplot(223)
    for i in aHO:
        plot_denstiy_aHO(20, i)
    plt.legend(['$a_{HO} = $' + str(i) + ' fm' for i in aHO])

    plt.subplot(224)
    for i in aHO:
        plot_denstiy_aHO(30, i)
    plt.legend(['$a_{HO} = $' + str(i) + ' fm' for i in aHO])

def logplot_denstiy_aHO(Nmax, aHO):
    x_model = np.linspace(0, 30, 500)
    
    plt.semilogy(x_model, np.abs(deuteron_wavefunc(Nmax, aHO, x_model)[3]))
    plt.ylim(bottom=10**(-6))
    plt.title('Aufenthaltswahrscheinlichkeit für $N_{max} = $' + str(Nmax) + ' mit logarithmischer y-Achse')
    plt.xlabel('r (fm)')

def logplots_density_aHO():
    plt.figure()
    
    plt.subplot(221)
    for i in aHO:
        logplot_denstiy_aHO(0, i)
    plt.legend(['$a_{HO} = $' + str(i) + ' fm' for i in aHO])
    
    plt.subplot(222)
    for i in aHO:
        logplot_denstiy_aHO(10, i)
    plt.legend(['$a_{HO} = $' + str(i) + ' fm' for i in aHO])

    plt.subplot(223)
    for i in aHO:
        logplot_denstiy_aHO(20, i)
    plt.legend(['$a_{HO} = $' + str(i) + ' fm' for i in aHO])

    plt.subplot(224)
    for i in aHO:
        logplot_denstiy_aHO(30, i)
    plt.legend(['$a_{HO} = $' + str(i) + ' fm' for i in aHO])

def logplots_density_rpow2_Nmax(aHO, NmaxMax, step):
    x_model_log = np.linspace(0, 30, 1000)
    
    plt.figure()
    
    for N in range(0, NmaxMax + 1, step):
        plt.semilogy(x_model_log, deuteron_wavefunc(N, aHO, x_model_log)[3] * x_model_log**2)
    plt.ylim(bottom=10**(-6), top=1)
    plt.title('$|\Phi(r)|^2 r^2$ bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$ mit logarithmischer y-Achse')
    plt.xlabel('r (fm)')
    plt.legend(['$N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)])
    plt.grid(linestyle='--')

def logplots_density_rpow4_Nmax(aHO, NmaxMax, step, xlim):
    x_model_log = np.linspace(0, 45, 1000)
    
    plt.figure()
    
    for N in range(0, NmaxMax + 1, step):
        plt.semilogy(x_model_log, deuteron_wavefunc(N, aHO, x_model_log)[3] * x_model_log**4)
    plt.ylim(bottom=10**(-6), top=10)
    plt.xlim(0,xlim)
    plt.title('$|\Phi(r)|^2 r^4$ bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$ mit logarithmischer y-Achse')
    plt.xlabel('r (fm)')
    plt.ylabel('$|\Phi(r)|^2 r^4$')
    plt.legend(['$N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)])
    plt.grid(linestyle='--')

def plots_density_rpow4_Nmax(aHO, NmaxMax, step, xlim):
    x_model_log = np.linspace(0, 45, 1000)
    
    plt.figure()
    
    for N in range(0, NmaxMax + 1, step):
        plt.plot(x_model_log, deuteron_wavefunc(N, aHO, x_model_log)[3] * x_model_log**4)
    # plt.ylim(bottom=10**(-6), top=10)
    plt.xlim(0,xlim)
    plt.title('$|\Phi(r)|^2 r^4$ bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$ mit logarithmischer y-Achse')
    plt.xlabel('r (fm)')
    plt.ylabel('$|\Phi(r)|^2 r^4$')
    plt.legend(['$N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)])
    plt.grid(linestyle='--')


def logplots_density_Nmax_lim(aHO, NNmax, NNmin, dNmax, xmin, xmax, ymin, ymax):
    x_model_log = np.arange(0, 70, 0.1)
    
    plt.figure()
    
    for N in range(NNmin, NNmax + dNmax, dNmax):
        plt.semilogy(x_model_log, deuteron_wavefunc(N, aHO, x_model_log)[3])
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$ mit logarithmischer y-Achse')
    plt.xlabel('r (fm)')
    plt.ylabel('$|\Phi(r)|^2$')
    plt.legend(['$N_{max} = $' + str(N) for N in range(0, NNmax + 1, dNmax)])
    plt.grid(linestyle='--')
    
# logplots_density_Nmax_lim(1.2, 150, 20, 10, 0, 42, 10**(-12), 1)
# logplots_density_Nmax_lim(1.4, 150, 20, 10, 0, 50, 10**(-14), 1)
# logplots_density_Nmax_lim(1.6, 150, 20, 10, 0, 55, 10**(-16), 1)
# logplots_density_Nmax_lim(1.8, 150, 20, 10, 0, 62, 10**(-16), 1)
# logplots_density_Nmax_lim(2.0, 150, 20, 10, 0, 70, 10**(-18), 1)

plots_wavefunc_Nmax(1.2, 60, 10)
# plots_wavefunc_Nmax(1.4, 60, 10)
# plots_wavefunc_Nmax(1.6, 60, 10)
# plots_wavefunc_Nmax(1.8, 60, 10)
# plots_wavefunc_Nmax(2.0, 60, 10)

# # plots_density_Nmax(1.2, 60, 10)
# plots_density_Nmax(1.4, 60, 5)
# plots_density_Nmax(1.4, 10, 1)
# # plots_density_Nmax(1.6, 60, 10)
# plots_density_Nmax(1.8, 60, 5)
# plots_density_Nmax(1.8, 10, 1)
# # plots_density_Nmax(2.0, 60, 10)

# logplots_density_Nmax(1.2, 10, 1, 12)
# logplots_density_Nmax(1.4, 10, 1, 13)
# logplots_density_Nmax(1.4, 60, 5, 17)
# logplots_density_Nmax(1.6, 10, 1, 15)
# logplots_density_Nmax(1.8, 10, 1, 16)
# logplots_density_Nmax(1.8, 60, 5, 18)
# logplots_density_Nmax(2.0, 10, 1, 17.5)

# # logplots_ddensity_dr_Nmax(1.2, 60, 5, 30)
# logplots_ddensity_dr_Nmax(1.4, 60, 5, 30)
# logplots_ddensity_dr_Nmax(1.4, 10, 1, 12)
# # logplots_ddensity_dr_Nmax(1.6, 60, 5, 35)
# logplots_ddensity_dr_Nmax(1.8, 60, 5, 40)
# logplots_ddensity_dr_Nmax(1.8, 10, 1, 18)
# # logplots_ddensity_dr_Nmax(2.0, 60, 5, 45)

# # plots_wavefunc_aHO()
# # plots_density_aHO()
# # logplots_density_aHO()

# # logplots_density_rpow2_Nmax(1.2, 60, 2)
# # logplots_density_rpow2_Nmax(1.4, 60, 2)
# # logplots_density_rpow2_Nmax(1.6, 60, 2)
# # logplots_density_rpow2_Nmax(1.8, 60, 2)
# # logplots_density_rpow2_Nmax(2.0, 60, 2)

# # logplots_density_rpow4_Nmax(1.2, 10, 1, 14)
# logplots_density_rpow4_Nmax(1.4, 10, 1, 12)
# logplots_density_rpow4_Nmax(1.4, 60, 5, 30)
# # logplots_density_rpow4_Nmax(1.6, 10, 1, 19)
# logplots_density_rpow4_Nmax(1.8, 60, 5, 40)
# logplots_density_rpow4_Nmax(1.8, 10, 1, 17)
# # logplots_density_rpow4_Nmax(2.0, 10, 1, 24)

# plots_density_rpow4_Nmax(1.4, 10, 1, 14)
# plots_density_rpow4_Nmax(1.8, 10, 1, 16)
# plots_density_rpow4_Nmax(1.4, 60, 5, 19)
# plots_density_rpow4_Nmax(1.8, 60, 5, 19)

plt.show()