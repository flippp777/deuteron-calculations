import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import sys

if sys.platform == "win32":
    sys.path.insert(1, r'C:\Users\phili.DESKTOP\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc
from plotstyle.colorFader import colorFader
from plotstyle.set_size import set_size



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
    plt.legend(['$\psi_0(r)$', '$\phi_{L=0}(r)$', '$\phi_{L=2}(r)$'])
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

# plots_wavefunc_Nmax(1.2, 60, 10)
# plots_wavefunc_Nmax(1.4, 60, 10)
# plots_wavefunc_Nmax(1.6, 60, 10)
# plots_wavefunc_Nmax(1.8, 60, 10)
# plots_wavefunc_Nmax(2.0, 60, 10

def final_wavefunc_plots(save=False):
    r_model = np.arange(0, 150, 0.1)
    Nmax = np.arange(0, 20, 2)
    
    purple = '#FF00FF'
    blue = '#0000FF'
    red = '#FF0000'
    black = '#000000'
    n = len(Nmax)
    
    fig = plt.figure(figsize=set_size(502.14963))
    gs = fig.add_gridspec(1, 2, wspace=0, hspace=0)
    axs = gs.subplots(sharey=True)
    
    def u(Nmax, aHO, r):
        if deuteron_wavefunc(Nmax, aHO, 2)[1] > 0:
            return r * deuteron_wavefunc(Nmax, aHO, r)[1]
        else:
            return -r * deuteron_wavefunc(Nmax, aHO, r)[1]

    def w(Nmax, aHO, r):
        if deuteron_wavefunc(Nmax, aHO, 2)[2] > 0:
            return r * deuteron_wavefunc(Nmax, aHO, r)[2]
        else:
            return -r * deuteron_wavefunc(Nmax, aHO, r)[2]

    def psi(Nmax, aHO, r):
        if deuteron_wavefunc(Nmax, aHO, 2)[0] > 0:
            return deuteron_wavefunc(Nmax, aHO, r)[0]
        else:
            return -deuteron_wavefunc(Nmax, aHO, r)[0]
    
    for i in range(n):
        axs[0].plot(r_model, psi(i, 1.2, r_model), color=colorFader(purple, black, i/n))
        axs[0].plot(r_model, u(i, 1.2, r_model), color=colorFader(blue, black, i/n))
        axs[0].plot(r_model, w(i, 1.2, r_model), color=colorFader(red, black, i/n))
    axs[0].set_xlim(0,10)
    axs[0].set_xlabel('$r$ (fm)')
    axs[0].set_ylabel('$\phi$')
    axs[0].grid(linestyle='--')

    for i in range(n):
        axs[1].plot(r_model, psi(i, 2, r_model), color=colorFader(purple, black, i/n))
        axs[1].plot(r_model, u(i, 2, r_model), color=colorFader(blue, black, i/n))
        axs[1].plot(r_model, w(i, 2, r_model), color=colorFader(red, black, i/n))
    # axs[1].semilogy()
    axs[1].set_xlim(0,20)
    axs[1].set_xlabel('$r$ (fm)')
    axs[1].grid(linestyle='--')
    
final_wavefunc_plots()

    
    


# plot_wavefunc_Nmax(60, 1.2)
# plot_wavefunc_Nmax(1.4, 60, 10)
# plot_wavefunc_Nmax(1.6, 60, 10)
# plot_wavefunc_Nmax(1.8, 60, 10)
# plot_wavefunc_Nmax(2.0, 60, 10)





plt.show()