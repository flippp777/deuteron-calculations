import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sys import exit

# Test lists
aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0]) 
Nmax_lst = np.arange(0, 62, 2)
r = np.linspace(0, 100, 1000)

from make_position_space_wavefunc import deuteron_wavefunc
def deut_density(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[3]

def derivative(x, f, n):
    dx = abs(x[0] - x[1])
    
    for i in range(n):
        f = np.gradient(f) / dx
        
    return f

def log_model(x, a, b):
    return a * x + b

def find_cont_par_cut(NNmax, dNmax, aHO):
    if NNmax < 3:
        exit("Nmax muss mindestens 3 sein")
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
    Nmax_lst = np.arange(2, NNmax + dNmax, dNmax)
    
    log_density = np.array([deut_density(r, 1, aHO)])
    with ThreadPoolExecutor() as executor:
       futures = [executor.submit(deut_density, r, Nmax, aHO) for Nmax in Nmax_lst]
       for future in futures:
           log_density = np.append(log_density, [future.result()], axis=0)
    log_density = np.log(log_density)
    
    r3_index = round(3/dr)
    diff_log_density_Nmax12 = np.abs(log_density[0, r3_index:] - log_density[1, r3_index:])
    start_linear_index = np.where(diff_log_density_Nmax12 == min(diff_log_density_Nmax12))[0][0] + r3_index 
    start_linear = r[start_linear_index]
    
    fit_points_indices = np.array([])
    rrms_lst = np.array([])
    slope_lst = np.array([])
    connection_lst = np.array([])
    y_intersect_lst = np.array([])
    norm_factor_lst = np.array([])
    
    for Nmax_i in range(2, NNmax, dNmax):
        for Nmax_j in range(1, Nmax_i, dNmax):
            diff_log_density = np.abs(log_density[Nmax_i, start_linear_index:] - log_density[Nmax_j, start_linear_index:])
            fit_points_indices = np.append(fit_points_indices, np.where(diff_log_density < 0.01)[0] + start_linear_index)
            if np.unique(fit_points_indices).size < 2:
                delta = 0.01 + 10**(-6)
                while np.unique(fit_points_indices).size < 2:
                    fit_points_indices = np.append(fit_points_indices, np.where(diff_log_density < delta)[0] + start_linear_index)
                    delta += 10**(-6)
              
        fit_points_indices = np.unique(fit_points_indices).astype(int)
        r_fit = r[fit_points_indices]
        log_density_fit = log_density[-1, fit_points_indices]
        
        slope = curve_fit(log_model, r_fit, log_density_fit)[0][0]
        
        d_dr = derivative(r, log_density[Nmax_i, start_linear_index:], 1)
        connection_point = np.array([])
        delta = 10**(-6)
        while connection_point.size == 0:
            connection_point = r[start_linear_index:][((d_dr > slope - delta) & (d_dr < slope + delta))]
            delta += 10**(-6)
        
        connection_lst = np.append(connection_lst, connection_point)
        
        y_intersect = np.log(deut_density(connection_point, Nmax_i, aHO)) - slope * connection_point
        

        r_wave = r[r <= connection_point]
        r_cont = r[r > connection_point]
        
        density_lst = deut_density(r_wave, Nmax_i, aHO)
        cont_lst = np.exp(log_model(r_cont, slope, y_intersect)) 
         
        norm_integ_wave = np.trapz(density_lst * r_wave**2, x=r_wave)
        norm_integ_cont = np.trapz(cont_lst * r_cont**2, x=r_cont)
        
        norm_integ = norm_integ_wave + norm_integ_cont
        
        norm_factor = 1 / norm_integ
        norm_factor_lst = np.append(norm_factor_lst, norm_factor)

        rsq_wave = np.trapz(density_lst * norm_factor * r_wave**4, x=r_wave) / 4
        rsq_cont = np.trapz(cont_lst * norm_factor * r_cont**4, x=r_cont) / 4
        
        rrms = np.sqrt(rsq_wave + rsq_cont)
        rrms_lst = np.append(rrms_lst, rrms)
        
        slope_lst = np.append(slope_lst, slope)
        y_intersect_lst = np.append(y_intersect_lst, y_intersect)
        
    rrms_lst = np.array([np.arange(3, NNmax + dNmax, dNmax), rrms_lst])  
    return slope_lst, y_intersect_lst, connection_lst, rrms_lst, norm_factor_lst







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


def test_plots(r, NNmax, dNmax):
    Nmax_lst = np.arange(2, NNmax + dNmax, dNmax)
    
    for j in range(aHO_lst.size):
        i = 0
        rrms = []
        plt.figure()
        plt.subplot(121)
        slope , y_sect, connect, rrms, norm = find_cont_par_cut(NNmax, dNmax, aHO_lst[j])
        r_cont = np.array([connect[i], 20])

        for Nmax in Nmax_lst:
            plt.plot(r, deut_density(r, Nmax, aHO_lst[j]) * norm[i], 'b')
            plt.plot(r_cont, np.exp(log_model(r_cont, slope[i], y_sect[i])) * norm[i], "r")
            plt.semilogy()
            plt.grid(linestyle='--')
        
        i =+ 1
        
        plt.title("Wellenfunktionen und exponentielle Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 20)
        plt.legend(["Wellenfunktion", "Fortsetzung"])
        
        rmax, dr = max(r), r[1]-r[0]
        rrms_full_wave = rrms_integral_trapz(60, 1.4, 2, rmax, dr)
        plt.subplot(122)
        plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
        plt.plot(rrms[0], rrms[1], 'x-')
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und exponentieller Fortsetzung')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
        plt.grid(linestyle='--')



            
test_plots(r, 5, 1)


plt.show()