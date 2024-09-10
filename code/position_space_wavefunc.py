import numpy as np
from matplotlib import pyplot as plt
from make_position_wavefunc import deuteron_wavefunc
from scipy import integrate

print(integrate.quad(lambda x: deuteron_wavefunc(4, 1.4, x)[3], 0, np.inf))
print(integrate.quad(lambda x: deuteron_wavefunc(15, 1.4, x)[3], 0, np.inf))
print(integrate.quad(lambda x: deuteron_wavefunc(30, 1.4, x)[3], 0, np.inf))
print(integrate.quad(lambda x: deuteron_wavefunc(40, 1.4, x)[3], 0, np.inf))


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

def plots_wavefunc_aHO(aHO):
    x_model = np.linspace(0, 15, 500)
    
    plt.figure()
    
    # Nmax = 0
    wavefuncs_Nmax0 = deuteron_wavefunc(0, aHO, x_model)
    wavefunc_Nmax0, wavefunc_Nmax0_L0, wavefunc_Nmax0_L2 = wavefuncs_Nmax0[0], wavefuncs_Nmax0[1], wavefuncs_Nmax0[2]
    
    plt.subplot(221)
    plt.plot(x_model, wavefunc_Nmax0)
    plt.plot(x_model, wavefunc_Nmax0_L0)
    plt.plot(x_model, wavefunc_Nmax0_L2)
    plt.title('Wellenfunctionen bei $a_{HO} = $' + str(aHO) +' fm für $N_{max} = 0$')
    plt.legend(['$\Phi(r)$', '$\Phi_{L=0}(r)$', '$\Phi_{L=2}(r)$'])
    plt.xlabel('r/aHO')
    plt.grid(linestyle='--')

    # Nmax = 20
    wavefuncs_Nmax20 = deuteron_wavefunc(20, aHO, x_model)
    wavefunc_Nmax20, wavefunc_Nmax20_L0, wavefunc_Nmax20_L2 = wavefuncs_Nmax20[0], wavefuncs_Nmax20[1], wavefuncs_Nmax20[2]

    plt.subplot(222)
    plt.plot(x_model, wavefunc_Nmax20)
    plt.plot(x_model, wavefunc_Nmax20_L0)
    plt.plot(x_model, wavefunc_Nmax20_L2)
    plt.title('Wellenfunctionen bei $a_{HO} = $' + str(aHO) +' fm für $N_{max} = 20$')
    plt.legend(['$\Phi(r)$', '$\Phi_{L=0}(r)$', '$\Phi_{L=2}(r)$'])
    plt.xlabel('r/aHO')
    plt.grid(linestyle='--')
    
    # Nmax = 40
    wavefuncs_Nmax40 = deuteron_wavefunc(40, aHO, x_model)
    wavefunc_Nmax40, wavefunc_Nmax40_L0, wavefunc_Nmax40_L2 = wavefuncs_Nmax40[0], wavefuncs_Nmax40[1], wavefuncs_Nmax40[2]
    
    plt.subplot(223)
    plt.plot(x_model, wavefunc_Nmax40)
    plt.plot(x_model, wavefunc_Nmax40_L0)
    plt.plot(x_model, wavefunc_Nmax40_L2)
    plt.title('Wellenfunctionen bei $a_{HO} = $' + str(aHO) +' fm für $N_{max} = 40$')
    plt.legend(['$\Phi(r)$', '$\Phi_{L=0}(r)$', '$\Phi_{L=2}(r)$'])
    plt.xlabel('r/aHO')
    plt.grid(linestyle='--')
    
    # Nmax = 60
    wavefuncs_Nmax60 = deuteron_wavefunc(60, aHO, x_model)
    wavefunc_Nmax60, wavefunc_Nmax60_L0, wavefunc_Nmax60_L2 = wavefuncs_Nmax60[0], wavefuncs_Nmax60[1], wavefuncs_Nmax60[2]
    
    plt.subplot(224)
    plt.plot(x_model, wavefunc_Nmax60)
    plt.plot(x_model, wavefunc_Nmax60_L0)
    plt.plot(x_model, wavefunc_Nmax60_L2)
    plt.title('Wellenfunctionen bei $a_{HO} = $' + str(aHO) +' fm für $N_{max} = 60$')
    plt.legend(['$\Phi(r)$', '$\Phi_{L=0}(r)$', '$\Phi_{L=2}(r)$'])
    plt.xlabel('r/aHO')
    plt.grid(linestyle='--')
    
def plots_density_aHO(aHO):
    x_model = np.linspace(0, 15, 500)
    
    plt.figure()
    
    # Nmax = 0
    wavefunc_Nmax0 = deuteron_wavefunc(0, aHO, x_model)[0]
    density_Nmax0 = wavefunc_Nmax0 * wavefunc_Nmax0
    
    plt.subplot(221)
    plt.plot(x_model, density_Nmax0)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} = $' + str(aHO) +' fm für $N_{max} = 0$')
    plt.xlabel('r/aHO')
    plt.grid(linestyle='--')

    # Nmax = 20
    wavefunc_Nmax20 = deuteron_wavefunc(20, aHO, x_model)[0]
    density_Nmax20 = wavefunc_Nmax20 * wavefunc_Nmax20
    
    plt.subplot(222)
    plt.plot(x_model, density_Nmax20)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} = $' + str(aHO) +' fm für $N_{max} = 20$')
    plt.xlabel('r/aHO')
    plt.grid(linestyle='--')
    
    # Nmax = 40
    wavefunc_Nmax40 = deuteron_wavefunc(40, aHO, x_model)[0]
    density_Nmax40 = wavefunc_Nmax40 * wavefunc_Nmax40
    
    plt.subplot(223)
    plt.plot(x_model, density_Nmax40)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} = $' + str(aHO) +' fm für $N_{max} = 40$')
    plt.xlabel('r/aHO')
    plt.grid(linestyle='--')
    
    # Nmax = 60
    wavefunc_Nmax60 = deuteron_wavefunc(60, aHO, x_model)[0]
    density_Nmax60 = wavefunc_Nmax60 * wavefunc_Nmax60
    
    plt.subplot(224)
    plt.plot(x_model, density_Nmax60)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} = $' + str(aHO) +' fm für $N_{max} = 60$')
    plt.xlabel('r/aHO')
    plt.grid(linestyle='--')

def logplots_density_aHO(aHO):
    x_model_log = np.linspace(0, 30, 500)
    
    plt.figure()
    
    # Nmax = 0
    wavefunc_Nmax0_log = deuteron_wavefunc(0, aHO, x_model_log)[0]
    density_Nmax0_log = wavefunc_Nmax0_log * wavefunc_Nmax0_log
    
    plt.subplot(221)
    plt.semilogy(x_model_log, density_Nmax0_log)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} = $' + str(aHO) +'  für $N_{max} = 0$ mit logarithmischer x-Achse')
    plt.xlabel('r/aHO ')
    plt.grid(linestyle='--')

    # Nmax = 20
    wavefunc_Nmax20_log = deuteron_wavefunc(20, aHO, x_model_log)[0]
    density_Nmax20_log = wavefunc_Nmax20_log * wavefunc_Nmax20_log
    
    plt.subplot(222)
    plt.semilogy(x_model_log, density_Nmax20_log)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} = $' + str(aHO) +'  für $N_{max} = 20$ mit logarithmischer x-Achse')
    plt.xlabel('r/aHO ')
    plt.grid(linestyle='--')
    
    # Nmax = 40
    wavefunc_Nmax40_log = deuteron_wavefunc(40, aHO, x_model_log)[0]
    density_Nmax40_log = wavefunc_Nmax40_log * wavefunc_Nmax40_log
    
    plt.subplot(223)
    plt.semilogy(x_model_log, density_Nmax40_log)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} = $' + str(aHO) +'  für $N_{max} = 40$ mit logarithmischer x-Achse')
    plt.xlabel('r/aHO ')
    plt.grid(linestyle='--')
    
    # Nmax = 60
    wavefunc_Nmax60_log = deuteron_wavefunc(60, aHO, x_model_log)[0]
    density_Nmax60_log = wavefunc_Nmax60_log * wavefunc_Nmax60_log
    
    plt.subplot(224)
    plt.semilogy(x_model_log, density_Nmax60_log)
    plt.title('Aufenthaltswahrscheinlichkeit bei $a_{HO} = $' + str(aHO) +'  für $N_{max} = 60$ mit logarithmischer x-Achse')
    plt.xlabel('r/aHO ')
    plt.grid(linestyle='--')
    


# plots_wavefunc_aHO(1.2)
# plots_wavefunc_aHO(1.4)
# plots_wavefunc_aHO(1.6)
# plots_wavefunc_aHO(1.8)
# plots_wavefunc_aHO(2.0)

# plots_density_aHO(1.2)
# plots_density_aHO(1.4)
# plots_density_aHO(1.6)
# plots_density_aHO(1.8)
# plots_density_aHO(2.0)

# logplots_density_aHO(1.2)
# logplots_density_aHO(1.4)
# logplots_density_aHO(1.6)
# logplots_density_aHO(1.8)
# logplots_density_aHO(2.0)

# plt.show()