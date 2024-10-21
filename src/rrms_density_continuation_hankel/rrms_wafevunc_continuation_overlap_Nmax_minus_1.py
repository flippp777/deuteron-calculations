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

def phi0(r, Nmax, aHO):
    return np.abs(deuteron_wavefunc(Nmax, aHO, r)[1])

def phi2(r, Nmax, aHO):
    return np.abs(deuteron_wavefunc(Nmax, aHO, r)[2])

def lnh0_sq_model(r, kappa):
    return -2 * (kappa * r + np.log(kappa) + np.log(r))

def lnh2_sq_model(r, kappa):
    return 2 * np.log(1/(kappa * r) * np.exp(- kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2))

def lnh0_sq_model_c(r, kappa, c):
    return -2 * (kappa * r + np.log(kappa) + np.log(r)) + c

def lnh2_sq_model_c(r, kappa, c):
    return 2 * np.log(c/(kappa * r) * np.exp(- kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2))


def find_continuation(NNmax, aHO, norm=True):
    # NNmax has to be bigger than 3 because Nmax=2 is the firts referece point to check for same densities
    if NNmax < 3:
        exit("NNmax muss mindestens 3 sein / NNmin darf nicht kleiner als 3 sein")

    # Define r range
    dr = 0.1
    r = np.arange(0, 100 + dr, dr)
   
    # create first list with Nmax=1 and then compute all other lists in parallel. Nmax goes from 2 to the closest number below NNmax that can be reached in steps the size of dNmax
    phi0_lst = np.array([phi0(r, NNmax-1, aHO), phi0(r, NNmax, aHO)])
    phi2_lst = np.array([phi2(r, NNmax-1, aHO), phi2(r, NNmax, aHO)])

    log_phi0 = np.log(phi0_lst)
    log_phi2 = np.log(phi2_lst)
    
    
    # Finding the starting point of the fit range. Minimum in range (0, 4)
    ddr_log_phi0 = derivative(r, log_phi0[1], 1)
    r4_index = int(4/dr)
    start_fit_index_phi0 = np.where(ddr_log_phi0[:r4_index] == min(ddr_log_phi0[:r4_index]))[0][0]
    
    ddr_log_phi2 = derivative(r, log_phi2[1], 1)
    r4_index = int(4/dr)
    start_fit_index_phi2 = np.where(ddr_log_phi2[:r4_index] == min(ddr_log_phi2[:r4_index]))[0][0]
   
    # Difference between log density at Nmax_j to Nmax_i starting from the start point of the linear range
    diff_log_phi0 = np.abs(log_phi0[0, start_fit_index_phi0:] - log_phi0[1, start_fit_index_phi0:])
    diff_log_phi2 = np.abs(log_phi2[0, start_fit_index_phi2:] - log_phi2[1, start_fit_index_phi2:])
    
    # Indiexes are at the points, where the difference in log densities is smaller 0.1
    distance = 0.01
    fit_points_indices_phi0 = np.where(diff_log_phi0 < distance)[0] + start_fit_index_phi0
    fit_points_indices_phi2 = np.where(diff_log_phi2 < distance)[0] + start_fit_index_phi2
    
    # condition to have two differen points in order to fit something
    if np.unique(fit_points_indices_phi0).size < 2:
        delta = distance + 10**(-6)
        while np.unique(fit_points_indices_phi0).size < 2:
            fit_points_indices_phi0 = np.append(fit_points_indices_phi0, np.where(diff_log_phi0 < delta)[0] + start_fit_index_phi0)
            delta += 10**(-6)
    if np.unique(fit_points_indices_phi2).size < 2:
        delta = distance + 10**(-6)
        while np.unique(fit_points_indices_phi2).size < 2:
            fit_points_indices_phi2 = np.append(fit_points_indices_phi2, np.where(diff_log_phi2 < delta)[0] + start_fit_index_phi2)
            delta += 10**(-6)
    
    # Removing duplicates
    fit_points_indices_phi0 = np.unique(fit_points_indices_phi0).astype(int)
    fit_points_indices_phi2 = np.unique(fit_points_indices_phi2).astype(int)

    end_fit_index = max(fit_points_indices_phi0) + 1
    
    # # use only the fit_point_num number of points in the middle of the list
    # fit_point_num = 4
    # if fit_points_indices.size > fit_point_num:
    #     middle_index = int(fit_points_indices.size / 2)
    #     # two_third_index = int(fit_points_indices.size * 2 / 3)
    #     three_quater_index = middle_index + int(fit_points_indices.size / 4)
    #     fit_points_indices = fit_points_indices[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]

    
    # fit points from fit_pint_indexes
    r_fit_phi0 = r[fit_points_indices_phi0]
    log_phi0_fit = log_phi0[1, fit_points_indices_phi0]
    fit_points_phi0 = np.array([r_fit_phi0, log_phi0_fit])
    
    r_fit_phi2 = r[fit_points_indices_phi2]
    log_phi2_fit = log_phi2[1, fit_points_indices_phi2]
    fit_points_phi2 = np.array([r_fit_phi2, log_phi2_fit])

    # fitting the points to finde the slope
    kappa_phi0 = curve_fit(lnh0_sq_model, r_fit_phi0, log_phi0_fit, [0.2305])[0]
    kappa_phi2 = curve_fit(lnh2_sq_model, r_fit_phi2, log_phi2_fit, [0.2305])[0]
    
    # Finding a point that has the same slope as the fitted one 
    d_dr_log_phi0 = derivative(r, log_phi0[1, fit_points_indices_phi0], 1)
    lnh0 = lnh0_sq_model(r_fit_phi0, kappa_phi0)
    ddr_lnh0 = derivative(r_fit_phi0, lnh0, 1)
    connection_point_phi0 = np.array([])
    delta = 10**(-6)
    while connection_point_phi0.size == 0:
        connection_point_phi0 = r_fit_phi0[((d_dr_log_phi0 > ddr_lnh0 - delta) & (d_dr_log_phi0 < ddr_lnh0 + delta))]
        delta += 10**(-6)
    connection_point_phi0 = connection_point_phi0[0]
    
    d_dr_log_phi2 = derivative(r, log_phi2[1, fit_points_indices_phi2], 1)
    lnh2 = lnh2_sq_model(r_fit_phi2, kappa_phi2)
    ddr_lnh2 = derivative(r_fit_phi2, lnh2, 1)
    connection_point_phi2 = np.array([])
    delta = 10**(-6)
    while connection_point_phi2.size == 0:
        connection_point_phi2 = r_fit_phi2[((d_dr_log_phi2 > ddr_lnh2 - delta) & (d_dr_log_phi2 < ddr_lnh2 + delta))]
        delta += 10**(-6)
    connection_point_phi2 = connection_point_phi2[0]

    c_phi0 = np.log(phi0(connection_point_phi0, NNmax, aHO)) - lnh0_sq_model(connection_point_phi0, kappa_phi0)
    c_phi2 = np.log(phi2(connection_point_phi2, NNmax, aHO)) - lnh2_sq_model(connection_point_phi2, kappa_phi2)

    def phi0_cont(r, Nmax, aHO):
        cont = []
        for i in range(r.size):
            if r[i] <= connection_point_phi0:
                cont.append(phi0_lst[1,i])
            else:
                cont.append(np.exp(lnh0_sq_model_c(r[i], kappa_phi0, c_phi0))[0])
        return np.array(cont)

    def phi2_cont(r, Nmax, aHO):
        cont = []
        for i in range(r.size):
            if r[i] <= connection_point_phi2:
                cont.append(phi2_lst[1,i])
            else:
                cont.append(np.exp(lnh2_sq_model_c(r[i], kappa_phi2, c_phi2))[0])
        return np.array(cont)
        
    r_wave_phi0 = r[r <= connection_point_phi0]
    r_cont_phi0 = r[r > connection_point_phi0]
    r_wave_phi2 = r[r <= connection_point_phi2]
    r_cont_phi2 = r[r > connection_point_phi2]
    
    phi0_lst = phi0_cont(r, NNmax, aHO)
    phi2_lst = phi2_cont(r, NNmax, aHO)
    
    if norm:
        norm_integ = trapezoid((phi0_lst**2 + phi2_lst**2) * r**2, x=r)
        
        norm_factor = 1 / norm_integ

        rsq = trapezoid((phi0_lst**2 + phi2_lst**2) * r**2, x=r) / 4

        rrms = np.sqrt(rsq)

        return kappa_phi0, c_phi0, connection_point_phi0, kappa_phi2, c_phi2, connection_point_phi2, rrms, norm_factor, fit_points_phi0, fit_points_indices_phi2
        
    else:
        norm = trapezoid((phi0_lst**2 + phi2_lst**2) * r**2, x=r)
        
        rsq = trapezoid((phi0_lst**2 + phi2_lst**2) * r**2, x=r) / 4

        rrms = np.sqrt(rsq)

        return kappa_phi0, c_phi0, connection_point_phi0, rrms, fit_points_phi0
    
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

aHO_lst = np.array([1.2, 2.0]) 
r = np.linspace(0, 100, 1000)


def plot_density_continuation_rrms(r, NNmax, NNmin, dNmax):
    Nmax_lst = np.arange(NNmin, NNmax + 1, dNmax)
    
    for j in range(aHO_lst.size):
        plt.figure()
        plt.subplot(121)

        rrms_lst = []
        for Nmax in Nmax_lst:
            kappa_phi0, c_phi0, connect_phi0, kappa_phi2, c_phi2, connect_phi2, rrms, norm, fit_points_phi0, fit_points_phi2 = find_continuation(Nmax, aHO_lst[j])

            r_cont_phi0 = np.arange(connect_phi0, 20.1, 0.1)
            r_cont_phi2 = np.arange(connect_phi2, 20.1, 0.1)

            plt.plot(r, phi0(r, Nmax, aHO_lst[j]) * norm, 'b')
            plt.plot(r, phi2(r, Nmax, aHO_lst[j]) * norm, 'c')
            plt.plot(r_cont_phi0, np.exp(lnh0_sq_model_c(r_cont_phi0, kappa_phi0, c_phi0)) * norm, "r")
            plt.plot(r_cont_phi2, np.exp(lnh2_sq_model_c(r_cont_phi2, kappa_phi2, c_phi2)) * norm, "r")
            # plt.scatter(fit_points[0], np.exp(fit_points[1]))
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



        
plot_density_continuation_rrms(r, 60, 3, 10)


plt.show()