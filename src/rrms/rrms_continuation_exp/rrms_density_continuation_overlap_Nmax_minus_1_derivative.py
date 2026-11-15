import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from concurrent.futures import ThreadPoolExecutor
from sys import exit
import sys

if sys.platform == 'win32':
    sys.path.insert(1, r'C:\Users\Phili\OneDrive\Uni OneDrive\Bachelorthesis\deuteron-calculations\src')
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

def log_model(x, a, b):
    return a * x + b

def find_cont_par_cut(NNmax, aHO, norm=True):
    # NNmax has to be bigger than 3 because Nmax=2 is the firts referece point to check for same densities
    if NNmax < 3:
        exit("NNmax muss mindestens 3 sein / NNmin darf nicht kleiner als 3 sein")

    # Define r range
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
   
    # create first list with Nmax=1 and then compute all other lists in parallel. Nmax goes from 2 to the closest number below NNmax that can be reached in steps the size of dNmax
    density = np.array([deut_density(r, 1, aHO), deut_density(r, NNmax-1, aHO), deut_density(r, NNmax, aHO)])

    log_density = np.log(density)
    
    # Finding the starting point of the linear range. It is chosen as the closest point to Nmax=1 density after r=3
    r3_index = int(3/dr)
    diff_log_density_Nmax1 = np.abs(log_density[0, r3_index:] - log_density[-1, r3_index:])
    start_linear_index = max(np.where(diff_log_density_Nmax1 < min(diff_log_density_Nmax1) + 0.1)[0]) + r3_index + NNmax *3
   
    # Difference between log density at Nmax_j to Nmax_i starting from the start point of the linear range
    diff_log_density = np.abs(log_density[1, start_linear_index:] - log_density[2, start_linear_index:])
    
    # End of the range where the difference in the logarithmic density is smaller than 0.01
    distance = 0.01
    end_linear_index = np.array([])
    while len(end_linear_index) < 2:
        distance += distance + 10**(-6)
        end_linear_index = np.append(end_linear_index,np.where(diff_log_density < distance)[0] + start_linear_index).flatten()

    end_linear_index = int(max(end_linear_index))
    
    # finding the slope with the mean of the derivative
    slope = np.mean(derivative(r, log_density[2, start_linear_index: end_linear_index + 1], 1))

    fit_points = [r[start_linear_index: end_linear_index + 1], log_density[2,start_linear_index: end_linear_index + 1]]
    
    # Finding a point that has the same slope as the found one 
    d_dr = derivative(r, log_density[2, start_linear_index: end_linear_index + 1], 1)
    connection_point = np.array([])
    delta = 10**(-6)
    while connection_point.size == 0:
        connection_point = r[start_linear_index: end_linear_index + 1][((d_dr > slope - delta) & (d_dr < slope + delta))]
        delta += 10**(-6)
    connection_point = connection_point[0]
    
    # Finding the y intersection
    y_intersect = np.log(deut_density(connection_point, NNmax , aHO)) - slope * connection_point
    

    r_wave = r[r <= connection_point]
    r_cont = r[r > connection_point]
    
    density_lst = deut_density(r_wave, NNmax, aHO)
    cont_lst = np.exp(log_model(r_cont, slope, y_intersect))
    
    if norm:    
        norm_integ_wave = trapezoid(density_lst * r_wave**2, x=r_wave)
        norm_integ_cont = trapezoid(cont_lst * r_cont**2, x=r_cont)
        
        norm_integ = norm_integ_wave + norm_integ_cont
        
        norm_factor = 1 / norm_integ

        rsq_wave = trapezoid(density_lst * norm_factor * r_wave**4, x=r_wave) / 4
        rsq_cont = trapezoid(cont_lst * norm_factor * r_cont**4, x=r_cont) / 4

        rrms = np.sqrt(rsq_wave + rsq_cont)

        return slope, y_intersect, connection_point, rrms, norm_factor, fit_points
        
    else:
        rsq_wave = trapezoid(density_lst * r_wave**4, x=r_wave) / 4
        rsq_cont = trapezoid(cont_lst * r_cont**4, x=r_cont) / 4
        
        rrms = np.sqrt(rsq_wave + rsq_cont)

        return slope, y_intersect, connection_point, rrms, fit_points


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


plot_density_continuation_rrms(r, 40, 3, 5)
plt.show()