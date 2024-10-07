import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from scipy.integrate import trapezoid 

from  rrms_density_continuation_fixed_slope import find_cont_par_cut, deut_density, log_model

def rsq_integral_trapz(Nmax, aHO, rmax, dr):
    r_size = int(np.round(rmax / dr))
    r = np.linspace(0, rmax + 1, r_size)
    integr = deut_density(r, Nmax, aHO) * r**4
    return trapezoid(integr, r, dr) / 4      # Warum Faktor 4?


def rrms_integral_trapz(Nmax, Nmin, aHO, Nstep, rmax, dr):
    rrms = []
    Nmax_lst = np.arange(Nmin, Nmax + 1, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rsq_integral_trapz, Nmax, aHO, rmax, dr) for Nmax in Nmax_lst]
        for future in futures:
            rrms.append(future.result())
    
    rrms = np.sqrt(np.array(rrms))
    
    return np.array([Nmax_lst, rrms])

aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0]) 
r = np.linspace(0, 100, 1000)


def plot_wavefunc_continuation_rrms(r, NNmax, NNmin, dNmax):
    Nmax_lst = np.arange(NNmin, NNmax + 1, dNmax)
    
    for j in range(aHO_lst.size):
        plt.figure()
        plt.subplot(121)

        rrms_lst = []
        for Nmax in Nmax_lst:
            slope , y_sect, connect, rrms, norm, fit_points = find_cont_par_cut(Nmax, aHO_lst[j])
            r_cont = np.array([connect, 20])
            plt.plot(r, deut_density(r, Nmax, aHO_lst[j]) * norm, 'b')
            plt.plot(r_cont, np.exp(log_model(r_cont, slope, y_sect)) * norm, "r")
            # plt.scatter(fit_points[0], np.exp(fit_points[1]))
            plt.semilogy()
            plt.grid(linestyle='--')
            rrms_lst.append(rrms)
        
        plt.title("Wellenfunktionen und exponentielle Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 20)
        plt.legend(["Wellenfunktion", "Fortsetzung"])
        
        rmax, dr = max(r), r[1]-r[0]
        rrms_full_wave = rrms_integral_trapz(NNmax, NNmin-3, aHO_lst[j], dNmax, rmax, dr)
        plt.subplot(122)
        plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
        plt.plot(Nmax_lst, rrms_lst, 'x-')
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und exponentieller Fortsetzung')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
        plt.grid(linestyle='--')


def plot_norm_comparison(r, NNmax, NNmin, dNmax):
    for aHO in aHO_lst:
        Nmax_lst =  np.arange(NNmin, NNmax + 1, dNmax)

        rrms_lst_norm = []
        norm_lst = []
        rrms_lst_nonorm = []
        
        for Nmax in Nmax_lst:
            rrms, norm = find_cont_par_cut(Nmax, aHO)[3:5]

            rrms_lst_norm.append(rrms)
            norm_lst.append(norm)
        
            rrms_nonorm = find_cont_par_cut(Nmax, aHO, norm=False)[3]
            rrms_lst_nonorm.append(rrms_nonorm)
        
        deviation = np.abs(np.array(rrms_lst_nonorm) - np.array(rrms_lst_norm)) / np.array(rrms_lst_nonorm)

        plt.figure()
        plt.subplot(211)
        plt.plot(Nmax_lst, norm_lst, 'x-')
        plt.title("Norm der fortgesetzten Wellenfunktion für $a_{HO} = $" + str(aHO))
        plt.xlabel('$N_{max}$')
        plt.grid(linestyle="--")

        plt.subplot(212)
        plt.plot(Nmax_lst, rrms_lst_nonorm, 'x-')
        plt.plot(Nmax_lst, rrms_lst_norm, 'x-')
        plt.title('$R_{rms}$ aus nicht normierter und normierter forgesetzten Wellenfunktion für $a_{HO} = $' + str(aHO))
        plt.legend(["Ohne Normierung", 'Mit Normierung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
        plt.grid(linestyle='--')
   
        
plot_wavefunc_continuation_rrms(r, 60, 3, 5)
# plot_norm_comparison(r, 60, 3, 2)


plt.show()