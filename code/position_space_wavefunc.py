import numpy as np
from matplotlib import pyplot as plt
from make_position_wavefunc import deuteron_wavefunc
from scipy import integrate

# print(integrate.quad(lambda x: deuteron_wavefunc(4, 1.4, x)[3], 0, np.inf))
# print(integrate.quad(lambda x: deuteron_wavefunc(15, 1.4, x)[3], 0, np.inf))
# print(integrate.quad(lambda x: deuteron_wavefunc(30, 1.4, x)[3], 0, np.inf))
# print(integrate.quad(lambda x: deuteron_wavefunc(40, 1.4, x)[3], 0, np.inf))


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
    plt.xlabel('r/aHO')
    plt.grid(linestyle='--')

def plots_wavefunc_aHO(aHO, NmaxMax, step):
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
    plt.xlabel('r/aHO ')
    plt.legend(np.array([['$\Phi_{L=0}(r)$, $N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)], ['$\Phi_{L=2}(r)$, $N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)]]).flatten())
    plt.grid(linestyle='--')
    
def plots_density_aHO(aHO, NmaxMax, step):
    x_model = np.linspace(0, 15, 500)
    
    plt.figure()
    
    for N in range(0, NmaxMax + 1, step):
        plt.plot(x_model, deuteron_wavefunc(N, aHO, x_model)[3])
    plt.xlim(right=10)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$')
    plt.xlabel('r/aHO ')
    plt.legend(['$N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)])
    plt.grid(linestyle='--')

def logplots_density_aHO(aHO, NmaxMax, step):
    x_model_log = np.linspace(0, 15, 500)
    
    plt.figure()
    
    for N in range(0, NmaxMax + 1, step):
        plt.semilogy(x_model_log, deuteron_wavefunc(N, aHO, x_model_log)[3])
    plt.ylim(bottom=10**(-6))
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$ mit logarithmischer x-Achse')
    plt.xlabel('r/aHO ')
    plt.legend(['$N_{max} = $' + str(N) for N in range(0, NmaxMax + 1, step)])
    plt.grid(linestyle='--')

    
    


plots_wavefunc_aHO(1.2, 60, 10)
plots_wavefunc_aHO(1.4, 60, 10)
plots_wavefunc_aHO(1.6, 60, 10)
plots_wavefunc_aHO(1.8, 60, 10)
plots_wavefunc_aHO(2.0, 60, 10)

# plots_density_aHO(1.2, 60, 10)
# plots_density_aHO(1.4, 60, 10)
# plots_density_aHO(1.6, 60, 10)
# plots_density_aHO(1.8, 60, 10)
# plots_density_aHO(2.0, 60, 10)

# logplots_density_aHO(1.2, 60, 10)
# logplots_density_aHO(1.4, 60, 10)
# logplots_density_aHO(1.6, 60, 10)
# logplots_density_aHO(1.8, 60, 10)
# logplots_density_aHO(2.0, 60, 10)

plt.show()