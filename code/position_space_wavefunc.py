import numpy as np
from matplotlib import pyplot as plt
from make_position_space_wavefunc import deuteron_wavefunc
from scipy import integrate

# print('density integral aHO = 1.2')
# for i in range(0,60,10):
#     print(integrate.quad(lambda x: 2*deuteron_wavefunc(i, 1.2, x)[3], 0, np.inf))
# print('density integral aHO = 1.4')
# for i in range(0,60,10):
#     print(integrate.quad(lambda x: 2*deuteron_wavefunc(i, 1.4, x)[3], 0, np.inf))
# print('density integral aHO = 1.6')
# for i in range(0,60,10):
#     print(integrate.quad(lambda x: 2*deuteron_wavefunc(i, 1.6, x)[3], 0, np.inf))
# print('density integral aHO = 1.8')
# for i in range(0,60,10):
#     print(integrate.quad(lambda x: 2*deuteron_wavefunc(i, 1.8, x)[3], 0, np.inf))
# print('density integral aHO = 2.0')
# for i in range(0,60,10):
#     print(integrate.quad(lambda x: 2*deuteron_wavefunc(i, 2.0, x)[3], 0, np.inf))


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
    plt.xlim(right=10)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$')
    plt.xlabel('r (fm)')
    plt.legend(['$N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)])
    plt.grid(linestyle='--')

def logplots_density_Nmax(aHO, NmaxMax, step):
    x_model_log = np.linspace(0, 15, 500)
    
    plt.figure()
    
    for N in range(0, NmaxMax + 1, step):
        plt.semilogy(x_model_log, deuteron_wavefunc(N, aHO, x_model_log)[3])
    plt.ylim(bottom=10**(-6))
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$ mit logarithmischer y-Achse')
    plt.xlabel('r (fm)')
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


# plots_wavefunc_Nmax(1.2, 60, 10)
# plots_wavefunc_Nmax(1.4, 60, 10)
# plots_wavefunc_Nmax(1.6, 60, 10)
# plots_wavefunc_Nmax(1.8, 60, 10)
# plots_wavefunc_Nmax(2.0, 60, 10)

# plots_density_Nmax(1.2, 60, 10)
# plots_density_Nmax(1.4, 60, 10)
# plots_density_Nmax(1.6, 60, 10)
# plots_density_Nmax(1.8, 60, 10)
# plots_density_Nmax(2.0, 60, 10)

# logplots_density_Nmax(1.2, 60, 10)
# logplots_density_Nmax(1.4, 60, 10)
# logplots_density_Nmax(1.6, 60, 10)
# logplots_density_Nmax(1.8, 60, 10)
# logplots_density_Nmax(2.0, 60, 10)

# plots_wavefunc_aHO()
# plots_density_aHO()
# logplots_density_aHO()

plt.show()