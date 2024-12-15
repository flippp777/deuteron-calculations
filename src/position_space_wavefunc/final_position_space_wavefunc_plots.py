import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import sys
import os

if sys.platform == "win32":
    sys.path.insert(1, r'C:\Users\phili.DESKTOP\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc
from plotstyle.colorFader import colorFader
from plotstyle.set_size import set_size


plt.style.use('src\stylesheet.mplstyle')


def plot_wavefunc_Nmax(Nmax, aHO):
    x_model = np.linspace(0, 15, 500)
    
    plt.figure()
    
    # Nmax = 0
    wavefuncs_Nmax = deuteron_wavefunc(Nmax, aHO, x_model)
    wavefunc_Nmax, wavefunc_Nmax_L0, wavefunc_Nmax_L2 = wavefuncs_Nmax[0], wavefuncs_Nmax[1], wavefuncs_Nmax[2]

    plt.plot(x_model, wavefunc_Nmax)
    plt.plot(x_model, wavefunc_Nmax_L0)
    plt.plot(x_model, wavefunc_Nmax_L2)
    plt.title('Wellenfunctionen bei $a_\\text{HO} = $' + str(aHO) +' fm für $N_{max} = $' + str(Nmax))
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
    plt.title('Wellenfunktionen bei $a_\\text{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$')
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
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_\\text{HO} =$' + str(aHO) + ' und verschiedenen $N_{max}$ mit logarithmischer y-Achse')
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
    Nmax = np.arange(0, 51, 10)
    
    purple = '#FF00FF'
    dark_purple = '#330033'
    blue = '#4c4cFF'
    dark_blue = '#000033'
    red = '#FF0000'
    dark_red = '#330000'
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
    
    legend = ['$u(r)$ \([1/\\sqrt{\\text{fm}}]\)', '$w(r)$ \([1/\\sqrt{\\text{fm}}]\)']
    
    axs[0].plot(r_model, u(150, 1.2, r_model), color=blue)
    axs[0].plot(r_model, w(150, 1.2, r_model), color=red)
    axs[0].text(15, 0.35, '$a_\\text{HO}=\\SI{1.2}{\\femto\\m}$')
    axs[0].set_xlim(0,25)
    axs[0].set_xticks([0, 5, 10, 15, 20])
    axs[0].set_ylim(top=0.6)
    axs[0].legend(legend)
    axs[0].set_xlabel('$r$ [fm]')
    axs[0].grid(linestyle='--')

    axs[1].plot(r_model, u(150, 2, r_model), color=blue)
    axs[1].plot(r_model, w(150, 2, r_model), color=red)
    axs[1].text(15, 0.35, '$a_\\text{HO}=\\SI{2.0}{\\femto\\m}$')
    axs[1].set_xlim(0,25)
    # axs[1].set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
    axs[1].legend(legend)
    axs[1].set_xlabel('$r$ [fm]')
    axs[1].grid(linestyle='--')

    if save:
        plt.savefig('wave_functions_nmax150_aHO1.2-2.0.pdf', bbox_inches='tight')

    
    fig = plt.figure(figsize=set_size(502.15))
    gs = fig.add_gridspec(1, 2, wspace=0, hspace=0)
    axs = gs.subplots(sharey=True)

    for i in range(n):
        axs[0].plot(r_model, u(i, 1.2, r_model), color=colorFader(blue, dark_blue, i/n))
        axs[0].plot(r_model, w(i, 1.2, r_model), color=colorFader(red, dark_red, i/n))
    axs[0].text(6, 0.45, '$a_\\text{HO}=\\SI{1.2}{\\femto\\m}$')
    axs[0].set_xlim(0,10)
    axs[0].set_xticks([0, 2, 4, 6, 8])
    axs[0].set_ylim(top=0.7)
    axs[0].legend(legend)
    axs[0].set_xlabel('$r$ [fm]')
    axs[0].grid(linestyle='--')

    for i in range(n):
        axs[1].plot(r_model, u(i, 2, r_model), color=colorFader(blue, black, i/n))
        axs[1].plot(r_model, w(i, 2, r_model), color=colorFader(red, black, i/n))
    axs[1].text(10, 0.45, '$a_\\text{HO}=\\SI{2.0}{\\femto\\m}$')
    axs[1].set_xlim(0,17.5)
    # axs[1].set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
    axs[1].legend(legend)
    axs[1].set_xlabel('$r$ [fm]')
    axs[1].grid(linestyle='--')
    if save:
        plt.savefig('wave_functions_nmax0-30_aHO1.2-2.0.pdf', bbox_inches='tight')


    fig = plt.figure(figsize=set_size(502.15))
    gs = fig.add_gridspec(1, 2, wspace=0, hspace=0)
    axs = gs.subplots(sharey=True)

    legend = ['$|u(r)|$ \([1/\\sqrt{\\text{fm}}]\)', '$|w(r)|$ \([1/\\sqrt{\\text{fm}}]\)'] 
    
    for i in range(n):
        axs[0].plot(r_model, np.abs(u(i, 1.2, r_model)), color=colorFader(blue, dark_blue, i/n))
        axs[0].plot(r_model, np.abs(w(i, 1.2, r_model)), color=colorFader(red, dark_red, i/n))
    axs[0].text(7.5, 0.3, '$a_\\text{HO}=\\SI{1.2}{\\femto\\m}$')
    axs[0].set_xlim(0,12.5)
    axs[0].set_xticks([0, 2.5, 5, 7.5, 10])
    axs[0].legend(legend)
    axs[0].set_xlabel('$r$ [fm]')
    axs[0].grid(linestyle='--')

    for i in range(n):
        axs[1].plot(r_model, np.abs(u(i, 2, r_model)), color=colorFader(blue, black, i/n))
        axs[1].plot(r_model, np.abs(w(i, 2, r_model)), color=colorFader(red, black, i/n))
    axs[1].text(13.5, 0.3, '$a_\\text{HO}=\\SI{2.0}{\\femto\\m}$')
    axs[1].semilogy()
    axs[1].set_ylim(10**(-5),1)
    axs[1].set_xlim(0,22.5)
    axs[1].legend(legend)
    axs[1].set_xlabel('$r$ [fm]')
    axs[1].grid(linestyle='--')
    if save:
        plt.savefig('wave_functions_nmax0-30_aHO1.2-2.0_log.pdf', bbox_inches='tight')

final_wavefunc_plots(True)

    
    


# plot_wavefunc_Nmax(60, 1.2)
# plot_wavefunc_Nmax(1.4, 60, 10)
# plot_wavefunc_Nmax(1.6, 60, 10)
# plot_wavefunc_Nmax(1.8, 60, 10)
# plot_wavefunc_Nmax(2.0, 60, 10)





plt.show()