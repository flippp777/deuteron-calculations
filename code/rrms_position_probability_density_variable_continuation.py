import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Test lists
aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0]) 
Nmax_lst = np.arange(0, 62, 2)
r = np.linspace(0, 15, 500)

from make_position_space_wavefunc import deuteron_wavefunc
def deut_density(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[3]


def log_model(x, a, b):
    return a * x + b

def find_cont_par_cut(r, Nmax, aHO):
    r = np.arange(5, 16, 0.5)
    log_density = np.log(deut_density(r, Nmax, aHO))
    d_dr = np.gradient(log_density)
    d2_dr2 = np.gradient(d_dr)
    d2_dr2 = np.array([r, d2_dr2]).transpose()
    
    d2_dr2_min = d2_dr2[d2_dr2[:,1] == min(d2_dr2[:,1])]

    r2 = np.arange(0, d2_dr2_min[0,0] + 0.1, 0.1)
    r2 = r2[np.abs(np.gradient(np.gradient(np.log(deut_density(r2, Nmax, aHO))))) < 0.005]
    max_r2 = max(r2)
    
    fit_points = [max_r2, max_r2 + 0.01]
    popt = curve_fit(log_model, fit_points, np.log(deut_density(fit_points, Nmax, aHO)))[0]
    
    return popt, max_r2 + 0.005,

def norm_cont_density(r, Nmax, aHO):
    popt, cut = find_cont_par_cut(r, Nmax, aHO)
    
    if popt.size != 0:
        r_wave = r[r <= cut]
        r_cont = r[r > cut]
        
        norm_integ_wave = np.trapezoid(deut_density(r_wave, Nmax, aHO) * r_wave**2, r_wave)
        norm_integ_cont = np.trapezoid(np.exp(log_model(r_cont, popt[0], popt[1])) * r_cont**2, r_cont)
        
        norm_integ = norm_integ_wave + norm_integ_cont
        
        norm_factor = 1 / norm_integ
        
        return norm_factor
    else: 
        return np.array([])


def rrms_cont_denstity(r, Nmax, aHO):
    popt, cut = find_cont_par_cut(r, Nmax, aHO)
    if popt.size != 0:
        r_wave = r[r <= cut]
        r_cont = r[r > cut]
        
        density_lst = deut_density(r_wave, Nmax, aHO)
        cont_lst = np.exp(log_model(r_cont, popt[0], popt[1])) 
        
        norm_integ_wave = np.trapezoid(density_lst * r_wave**2, r_wave)
        norm_integ_cont = np.trapezoid(cont_lst * r_cont**2, r_cont)
        
        norm_integ = norm_integ_wave + norm_integ_cont
        norm_factor = 1 / norm_integ

        rsq_wave = np.trapezoid(density_lst * norm_factor, r_wave**4, r_wave) / 4
        rsq_cont = np.trapezoid(cont_lst * norm_factor, r_cont**4, r_cont) / 4
        
        rrms = np.sqrt(rsq_wave + rsq_cont)
        
        return rrms
    else: 
        return np.array([])





def rsq_integral_trapezoid(Nmax, aHO, rmax, dr):
    r_size = int(np.round(rmax / dr))
    r = np.linspace(0, rmax + 1, r_size)
    integr = deut_density(r, Nmax, aHO) * r**4
    return np.trapezoid(integr, r, dr) / 4      # Warum Faktor 4?


def rrms_integral_trapezoid(Nmax, aHO, Nstep, rmax, dr):
    rrms = []
    Nmax_lst = np.arange(0, Nmax + 1, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rsq_integral_trapezoid, Nmax, aHO, rmax, dr) for Nmax in Nmax_lst]
        for future in futures:
            rrms.append(future.result())
    
    rrms = np.sqrt(np.array(rrms))
    
    return np.array([Nmax_lst, rrms])


def test_plots(r, NNmax, dNmax):
    Nmax_lst = np.arange(2, NNmax + dNmax, dNmax)
    
    for j in range(aHO_lst.size):
        rrms = []
        plt.figure()
        plt.subplot(121)
        for Nmax in Nmax_lst:
            popt , cut = find_cont_par_cut(r, Nmax, aHO_lst[j])
            r_cont = np.array([cut, 20])
            norm = norm_cont_density(r, Nmax, aHO_lst[j])
            
            plt.plot(r, deut_density(r, Nmax, aHO_lst[j]) * norm, 'b')
            plt.plot(r_cont, np.exp(log_model(r_cont, popt[0], popt[1])) * norm, "r")
            plt.semilogy()
            plt.grid(linestyle='--')
            
            rrms.append(rrms_cont_denstity(r, Nmax, aHO_lst[j]))
            
        plt.title("Wellenfunktionen und exponentielle Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 15)
        plt.legend(["Wellenfunktion", "Fortsetzung"])
        
        rmax, dr = max(r), r[1]-r[0]
        rrms_full_wave = rrms_integral_trapezoid(60, 1.4, 2, rmax, dr)
        plt.subplot(122)
        plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
        plt.plot(Nmax_lst, rrms, 'x-')
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(1.4) + ' fm aus Wellenfunktion und exponentieller Fortsetzung bei r = 4 fm')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
        plt.grid(linestyle='--')


def test_d(r, NNmax, dNmax):
    Nmax_lst = np.arange(1, NNmax + dNmax, dNmax)
    
    for j in range(aHO_lst.size):
        plt.figure()
        for Nmax in Nmax_lst:
            plt.plot(r, np.log(deut_density(r, Nmax, aHO_lst[j])), 'b')
            plt.plot(r, np.gradient(np.log(deut_density(r, Nmax, aHO_lst[j]))) * 100, 'r')
            plt.plot(r, np.gradient(np.gradient(np.log(deut_density(r, Nmax, aHO_lst[j])))) * 10000, 'g')
            plt.title('aHO = ' + str(aHO_lst[j]))
            plt.grid(linestyle='--')
            

# test_plots(r, 6, 2)

test_d(r, 6, 6)

plt.show()