import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sys import exit
import sys

if sys.platform == 'win32':
    sys.path.insert(1, r'C:\Users\Phili\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc
def deut_density(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[3]

def derivative(x, f, n):
    dx = abs(x[0] - x[1])
    
    for i in range(n):
        f = np.gradient(f) / dx
        
    return f


def lnh0_model(r, kappa, c):
    return -2 * (kappa * r + np.log(kappa) + np.log(r)) + c


def find_continuation(NNmax, aHO, norm=True):
    # NNmax has to be bigger than 3 because Nmax=2 is the firts referece point to check for same densities
    if NNmax < 3:
        exit("NNmax muss mindestens 3 sein / NNmin darf nicht kleiner als 3 sein")

    # Define r range
    dr = 0.1
    r = np.arange(0, 100 + dr, dr)
   
    # create first list with Nmax=1 and then compute all other lists in parallel. Nmax goes from 2 to the closest number below NNmax that can be reached in steps the size of dNmax
    density = np.array([deut_density(r, NNmax-1, aHO), deut_density(r, NNmax, aHO)])

    log_density = np.log(density)
    
    
    # Finding the starting point of the fit range. Minimum in range (0, 4)
    ddr_log_density = derivative(r, log_density[1], 1)
    r4_index = int(4/dr)
    start_fit_index = np.where(ddr_log_density[:r4_index] == min(ddr_log_density[:r4_index]))[0][0] + 10
    
   
    # Difference between log density at Nmax_j to Nmax_i starting from the start point of the linear range
    diff_log_density = np.abs(log_density[0, start_fit_index:] - log_density[1, start_fit_index:])
    
    # Indiexes are at the points, where the difference in log densities is smaller 0.1
    distance = 0.01
    fit_points_indices = np.where(diff_log_density < distance)[0] + start_fit_index
    
    # condition to have two differen points in order to fit something
    if np.unique(fit_points_indices).size < 2:
        delta = distance + 10**(-6)
        while np.unique(fit_points_indices).size < 2:
            fit_points_indices = np.append(fit_points_indices, np.where(diff_log_density < delta)[0] + start_fit_index)
            delta += 10**(-6)
    
    # Removing duplicates
    fit_points_indices = np.unique(fit_points_indices).astype(int)

    end_fit_index = max(fit_points_indices) + 1
    
    # # use only the fit_point_num number of points in the middle of the list
    # fit_point_num = 4
    # if fit_points_indices.size > fit_point_num:
    #     middle_index = int(fit_points_indices.size / 2)
    #     # two_third_index = int(fit_points_indices.size * 2 / 3)
    #     three_quater_index = middle_index + int(fit_points_indices.size / 4)
    #     fit_points_indices = fit_points_indices[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]

    
    # fit points from fit_pint_indexes
    r_fit = r[start_fit_index:end_fit_index]
    log_density_fit = log_density[1, start_fit_index:end_fit_index]
    fit_points = np.array([r_fit, log_density_fit])
    
    # fitting the points to finde the slope
    kappa, c = curve_fit(lnh0_model, r_fit, log_density_fit, [0.2305, 0])[0]
    
    # Finding a point that has the same slope as the fitted one 
    d_dr_log_densiy = derivative(r, log_density[1, start_fit_index:end_fit_index], 1)
    lnh0 = lnh0_model(r_fit, kappa, c)
    ddr_lnh0 = derivative(r_fit, lnh0, 1)
    connection_point = np.array([])
    delta = 10**(-6)
    while connection_point.size == 0:
        connection_point = r_fit[((log_density_fit > lnh0 - delta) & (log_density_fit < lnh0 + delta) & (d_dr_log_densiy > ddr_lnh0 - delta) & (d_dr_log_densiy < ddr_lnh0 + delta))]
        delta += 10**(-6)
    connection_point = connection_point[0]
    

    r_wave = r[r <= connection_point]
    r_cont = r[r > connection_point]
    
    density_lst = deut_density(r_wave, NNmax, aHO)
    cont_lst = np.exp(lnh0_model(r_cont, kappa, c))
    
    if norm:    
        norm_integ_wave = trapezoid(density_lst * r_wave**2, x=r_wave)
        norm_integ_cont = trapezoid(cont_lst * r_cont**2, x=r_cont)
        
        norm_integ = norm_integ_wave + norm_integ_cont
        
        norm_factor = 1 / norm_integ

        rsq_wave = trapezoid(density_lst * norm_factor * r_wave**4, x=r_wave) / 4
        rsq_cont = trapezoid(cont_lst * norm_factor * r_cont**4, x=r_cont) / 4

        rrms = np.sqrt(rsq_wave + rsq_cont)

        return kappa, c, connection_point, rrms, norm_factor, fit_points
        
    else:
        rsq_wave = trapezoid(density_lst * r_wave**4, x=r_wave) / 4
        rsq_cont = trapezoid(cont_lst * r_cont**4, x=r_cont) / 4
        
        rrms = np.sqrt(rsq_wave + rsq_cont)

        return kappa, c, connection_point, rrms, fit_points
    
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


def plot_density_continuation_rrms(r, NNmax, NNmin, dNmax):
    Nmax_lst = np.arange(NNmin, NNmax + 1, dNmax)
    
    for j in range(aHO_lst.size):
        plt.figure()
        plt.subplot(121)

        rrms_lst = []
        for Nmax in Nmax_lst:
            kappa, c, connect, rrms, norm, fit_points = find_continuation(Nmax, aHO_lst[j])
            r_cont = np.array([connect, 20])
            plt.plot(r, deut_density(r, Nmax, aHO_lst[j]) * norm, 'b')
            plt.plot(r_cont, np.exp(lnh0_model(r_cont, kappa, c)) * norm, "r")
            plt.scatter(fit_points[0], np.exp(fit_points[1]))
            plt.semilogy()
            plt.xlabel('r (fm)')
            plt.ylabel('$|\Phi(r)|^2$')
            plt.grid(linestyle='--')
            rrms_lst.append(rrms)
        
        plt.title("Wellenfunktionen und Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 20)
        plt.legend(["Wellenfunktion", "Fortsetzung"])
        
        rmax, dr = max(r), r[1]-r[0]
        rrms_full_wave = rrms_integral_trapz(NNmax, NNmin-3, aHO_lst[j], dNmax, rmax, dr)
        plt.subplot(122)
        plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
        plt.plot(Nmax_lst, rrms_lst, 'x-')
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und Fortsetzung')
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
            rrms, norm = find_continuation(Nmax, aHO)[3:5]

            rrms_lst_norm.append(rrms)
            norm_lst.append(norm)
        
            rrms_nonorm = find_continuation(Nmax, aHO, norm=False)[3]
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
   
        
plot_density_continuation_rrms(r, 60, 3, 2)


plt.show()