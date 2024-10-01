import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import quad
from scipy.optimize import curve_fit

from make_position_space_wavefunc import deuteron_wavefunc


aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0])

def rsq_integral_trapz(Nmax, aHO, rmax, dr):
    r_size = int(np.round(rmax / dr))
    r = np.linspace(0, rmax + 1, r_size)
    integr = deuteron_wavefunc(Nmax, aHO, r)[3] * r**4
    return np.trapz(integr, r, dr) / 4      # Warum Faktor 4?

def rrms_integral_trapz(Nmax, aHO, Nstep, rmax, dr):
    rrms = []
    Nmax_lst = np.arange(0, Nmax + 1, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rsq_integral_trapz, Nmax, aHO, rmax, dr) for Nmax in Nmax_lst]
        for future in futures:
            rrms.append(future.result())
    
    rrms = np.sqrt(np.array(rrms))
    
    return np.array([Nmax_lst, rrms])

def lin_log_model(x, a, b):
    return x * a + b

def cont_at4(Nmax, aHO, cut):

    r_fit = [cut - 0.01, cut] 
    logy_fit = np.log(deuteron_wavefunc(Nmax, aHO, r_fit)[3])
    popt = curve_fit(lin_log_model, r_fit, logy_fit)[0]

    return popt

def norm_cont(Nmax, aHO, cut, rmax, dr):
    r_wave = np.arange(0, cut + dr, dr)
    r_cont = np.arange(cut, rmax + dr, dr)
    
    int_wave = np.trapz(deuteron_wavefunc(Nmax, aHO, r_wave)[3] * r_wave**2, x=r_wave)
    
    cont_par = cont_at4(Nmax, aHO, cut)
    int_cont = np.trapz(np.exp(lin_log_model(r_cont, cont_par[0], cont_par[1])) * r_cont**2, x=r_cont)
    
    integral = int_wave + int_cont
    
    norm_factor = 1 / integral
        
    return norm_factor

def rrms_int_cont_at4(Nmax, aHO, cut, rmax, dr):
    r_wave = np.arange(0, cut + dr, dr)
    r_cont = np.arange(cut, rmax + dr, dr)

    norm = norm_cont(Nmax, aHO, cut, rmax, dr)
    
    rsq_wave = np.trapz(deuteron_wavefunc(Nmax, aHO, r_wave)[3] * norm_cont(Nmax, aHO, cut, rmax, dr) * r_wave**4 * norm, x=r_wave) / 4
    
    cont_par = cont_at4(Nmax, aHO, cut)
    rsq_cont = np.trapz(np.exp(lin_log_model(r_cont, cont_par[0], cont_par[1])) * norm_cont(Nmax, aHO, cut, rmax, dr) * r_cont**4 * norm, x=r_cont) / 4
    
    rrms = np.sqrt(rsq_wave + rsq_cont)
    
    return rrms
    
def first_plots(NNmax, dNmax, rmax, dr, cut):
    Nmax_lst = np.arange(0, NNmax + dNmax, dNmax)
    r_wave = np.arange(0, 4 + dr, dr)
    r_cont = np.arange(cut, rmax + dr, dr)
    r_full_plot = np.arange(0, 15 + dr, dr)

    for j in range(aHO_lst.size):
        rrms = []
        plt.figure()
        plt.subplot(121)
        legend = []
        for Nmax in Nmax_lst:
            cont_par = cont_at4(Nmax, aHO_lst[j], cut)
            plt.semilogy(r_full_plot, deuteron_wavefunc(Nmax, aHO_lst[j], r_full_plot)[3], "b")
            plt.semilogy(r_cont, np.exp(lin_log_model(r_cont, cont_par[0], cont_par[1])), "r")
            legend.append('$N_{max} = $' + str(Nmax) + ', Probabiliy densitz')
            legend.append('$N_{max} = $' + str(Nmax) + ', Continuation')
            
            rrms.append(rrms_int_cont_at4(Nmax, aHO_lst[j], cut, 100, 0.1))
        
        plt.title("Wellenfunktionen und exponentielle Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 15)
        plt.legend(["Wellenfunktion", "Fortsetzung"])
        
        rrms_full_wave = rrms_integral_trapz(NNmax, aHO_lst[j], dNmax, rmax, dr)
        plt.subplot(122)
        plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
        plt.plot(Nmax_lst, rrms, 'x-')
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und exponentieller Fortsetzung bei r = 4 fm')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
        plt.grid(linestyle='--')

first_plots(60, 2, 100, 0.1, 5)

plt.show()