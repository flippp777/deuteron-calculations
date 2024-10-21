import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import trapezoid
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

def h0_sq_model(r, kappa):
    return 1 / (kappa * r)**2 * np.exp(-2 * kappa * r)

def h0_sq_model_a(r, a, kappa):
    return a / (kappa * r)**2 * np.exp(-2 * kappa * r)


kappa = 0.2307

def find_cont_par_cut(NNmax, aHO, norm=True):
    # NNmax has to be bigger than 3 because Nmax=2 is the firts referece point to check for same densities
    if NNmax < 3:
        sys.exit("NNmax muss mindestens 3 sein / NNmin darf nicht kleiner als 3 sein")

    # Define r range
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
   
    # create first list with Nmax=1 and then compute all other lists in parallel. Nmax goes from 2 to the closest number below NNmax that can be reached in steps the size of dNmax
    density = np.array([deut_density(r, 1, aHO), deut_density(r, NNmax-1, aHO), deut_density(r, NNmax, aHO)])

    log_density = np.log(density)
    
    # Finding the starting point of the exponential range. It is chosen as the closest point to Nmax=1 density after r=3
    r3_index = int(3/dr)
    diff_log_density_Nmax1 = np.abs(log_density[0, r3_index:] - log_density[-1, r3_index:])
    start_linear_index = max(np.where(diff_log_density_Nmax1 < min(diff_log_density_Nmax1) + 0.1)[0]) + r3_index
    
    # Difference between log density at Nmax_j to Nmax_i starting from the start point of the linear range
    diff_log_density = np.abs(log_density[1, start_linear_index:] - log_density[2, start_linear_index:])

    # Indiexes are at the points, where the difference in log densities is smaller 0.1
    distance = 0.01
    fit_indices = np.where(diff_log_density < distance)[0] + start_linear_index
    
    # Removing duplicates
    fit_indices = np.unique(fit_indices).astype(int)

    # condition to have two differen points in order to fit a straight
    if np.unique(fit_indices).size < 20:
        delta = distance + 10**(-3)
        while np.unique(fit_indices).size < 20:
            fit_indices = np.append(fit_indices, np.where(diff_log_density < delta)[0] + start_linear_index)
            delta += 10**(-3)
    
    # Removing duplicates
    fit_indices = np.unique(fit_indices).astype(int)
    
    # # use only the fit_point_num number of points in the middle of the list
    connect_point_num = 10
    if fit_indices.size > connect_point_num:
        middle_index = int(fit_indices.size / 2)
        two_third_index = int(fit_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_indices.size / 4)
        fit_indices = fit_indices[-connect_point_num:]
    
    
    # Finding a point that has the same slope as the fitted one 
    d_dr_density = derivative(r, density[2, fit_indices], 1)
    h0_sq = h0_sq_model(r[fit_indices], kappa)
    d_dr_h0_sq = derivative(r[fit_indices], h0_sq, 1)

    connection_point = np.array([])
    delta = 10**(-6)
    while connection_point.size == 0:
        connection_point = r[fit_indices][((d_dr_density > d_dr_h0_sq - delta) & (d_dr_density < d_dr_h0_sq + delta))]
        delta += 10**(-6)
    connection_point = connection_point[0]
    
    # Finding the factor so that h0 fits on the curve
    a = deut_density(connection_point, NNmax, aHO) / h0_sq_model(connection_point, kappa)
    
    r_wave = r[r <= connection_point]
    r_cont = r[r > connection_point]
    
    density_lst = deut_density(r_wave, NNmax, aHO)
    cont_lst = h0_sq_model_a(r_cont, a, kappa)
    
    if norm:    
        norm_integ_wave = trapezoid(density_lst * r_wave**2, x=r_wave)
        norm_integ_cont = trapezoid(cont_lst * r_cont**2, x=r_cont)
        
        norm_integ = norm_integ_wave + norm_integ_cont
        
        norm_factor = 1 / norm_integ

        rsq_wave = trapezoid(density_lst * norm_factor * r_wave**4, x=r_wave) / 4
        rsq_cont = trapezoid(cont_lst * norm_factor * r_cont**4, x=r_cont) / 4

        rrms = np.sqrt(rsq_wave + rsq_cont)

        return a, kappa, connection_point, rrms, norm_factor, np.array([r[fit_indices], density[2, fit_indices]])
        
    else:
        rsq_wave = trapezoid(density_lst * r_wave**4, x=r_wave) / 4
        rsq_cont = trapezoid(cont_lst * r_cont**4, x=r_cont) / 4
        
        rrms = np.sqrt(rsq_wave + rsq_cont)

        return a, kappa, connection_point, rrms, np.array([r[fit_indices], density[2, fit_indices]])

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
            a, kappa, connect, rrms, norm, fit_points = find_cont_par_cut(Nmax, aHO_lst[j])
            r_cont = r[r>connect]
            plt.plot(r, deut_density(r, Nmax, aHO_lst[j]) * norm, 'b')
            plt.plot(r_cont, h0_sq_model_a(r_cont, a, kappa) * norm, "r")
            # plt.scatter(fit_points[0], np.exp(fit_points[1]))
            plt.semilogy()
            plt.grid(linestyle='--')
            rrms_lst.append(rrms)
        
        plt.title("Wellenfunktionen und  Fortzetzung mit $\kappa = $" + str(kappa) + " 1/fm für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
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


   
        
plot_wavefunc_continuation_rrms(r, 60, 3, 2)


plt.show()