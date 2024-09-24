import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

from rrms_position_probability_density_continuation_variable import deut_density, log_model, log_cont_par_and_cut, rrms_cont_denstity, find_fit_log_points_and_cut, norm_cont_density

aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
Nmax_lst = np.arange(2, 62, 2)
r_model = np.linspace(0,20, 500)

def rsq_integral_trapz(Nmax, aHO, rmax, dr):
    r_size = int(np.round(rmax / dr))
    r = np.linspace(0, rmax + 1, r_size)
    integr = deut_density(r, Nmax, aHO) * r**4
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

def Nmax_calc(r, Nmax, aHO, d_d2_dr2):
    popt, cut = log_cont_par_and_cut(r, Nmax, aHO, d_d2_dr2)

    rrms =  rrms_cont_denstity(r, Nmax, aHO, d_d2_dr2)
    
    return popt, cut, rrms

def aHO_calc(r, Nmax_lst, aHO, d_d2_dr2):
    popt_cut = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(log_cont_par_and_cut, r, Nmax, aHO, d_d2_dr2) for Nmax in Nmax_lst]
        for future in futures:
            popt_cut.append(future.result())
    
    return popt_cut
            
def first_plots(r, NNmax, dNmax, d_d2dr2):
    Nmax_lst = np.arange(0, NNmax + dNmax, dNmax)

    for j in range(aHO_lst.size):
        rrms = []
        plt.figure()
        plt.subplot(121)
        legend = []
        for Nmax in Nmax_lst:
            popt , cut = log_cont_par_and_cut(r, Nmax, aHO_lst[j], d_d2dr2)
            if popt.size != 0:
                r_cont = r[r > cut]
                fit_points = find_fit_log_points_and_cut(r, Nmax, aHO_lst[j], d_d2dr2)[0]
                
                norm = norm_cont_density(r, Nmax, aHO_lst[j], d_d2dr2)
                
                plt.semilogy(r, deut_density(r, Nmax, aHO_lst[j]) * norm, "b")
                plt.semilogy(r_cont, np.exp(log_model(r_cont, popt[0], popt[1])) * norm, "r")
                plt.scatter(fit_points[:,0], fit_points[:,1])
                
                legend.append('$N_{max} = $' + str(Nmax) + ', Probabiliy densitz')
                legend.append('$N_{max} = $' + str(Nmax) + ', Continuation')
                
                rrms.append(rrms_cont_denstity(r, Nmax, aHO_lst[j], d_d2dr2))
            else:
                rrms.append(0)
        
        plt.title("Wellenfunktionen und exponentielle Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 15)
        plt.legend(["Wellenfunktion", "Fortsetzung"])

        rrms_full_wave = rrms_integral_trapz(NNmax, aHO_lst[j], dNmax, 100, 0.1)
        plt.subplot(122)
        plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
        plt.plot(Nmax_lst, rrms, 'x-')
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und exponentieller Fortsetzung bei r = 4 fm')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
        plt.grid(linestyle='--')

first_plots(r_model, 60, 2, 10**(-4))

plt.show()

