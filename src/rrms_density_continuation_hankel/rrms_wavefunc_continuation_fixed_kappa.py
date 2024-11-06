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

plt.style.use('tableau-colorblind10')
def deut_density(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[3]

def phi0(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[1]**2

def phi2(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[2]**2

def derivative(x, f, n):
    dx = abs(x[0] - x[1])
    
    for i in range(n):
        f = np.gradient(f) / dx
        
    return f

def h0_sq_model(r, kappa):
    return 1 / (kappa * r)**2 * np.exp(-2 * kappa * r)

def h0_sq_model_a(r, a, kappa):
    return a / (kappa * r)**2 * np.exp(-2 * kappa * r)

def h2_sq_model(r, kappa):
    return 1/(kappa * r)**2 * np.exp(-2 * kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2)**2

def h2_sq_model_a(r, a, kappa):
    return a/(kappa * r)**2 * np.exp(-2 * kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2)**2

kappa = 0.2307 # kappa from real energy
# kappa = 0.2171 # kappa from found energy

def find_cont_par_cut(NNmax, aHO):
    # NNmax has to be bigger than 3 because Nmax=2 is the firts referece point to check for same densities
    if NNmax < 3:
        sys.exit("NNmax muss mindestens 3 sein / NNmin darf nicht kleiner als 3 sein")

    # Define r range
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
   
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
        delta = distance + 10**(-3)
        while np.unique(fit_points_indices_phi0).size < 2:
            fit_points_indices_phi0 = np.append(fit_points_indices_phi0, np.where(diff_log_phi0 < delta)[0] + start_fit_index_phi0)
            delta += 10**(-3)
    if np.unique(fit_points_indices_phi2).size < 2:
        delta = distance + 10**(-3)
        while np.unique(fit_points_indices_phi2).size < 2:
            fit_points_indices_phi2 = np.append(fit_points_indices_phi2, np.where(diff_log_phi2 < delta)[0] + start_fit_index_phi2)
            delta += 10**(-3)
    
    # Removing duplicates
    fit_points_indices_phi0 = np.unique(fit_points_indices_phi0).astype(int)
    fit_points_indices_phi2 = np.unique(fit_points_indices_phi2).astype(int)

    end_fit_index = max(fit_points_indices_phi0) + 1
    
    # use only the fit_point_num number of points in the middle of the list
    fit_point_num = 10
    if fit_points_indices_phi0.size > fit_point_num:
        middle_index = int(fit_points_indices_phi0.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices_phi0.size / 4)
        fit_points_indices_phi0 = fit_points_indices_phi0[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]
    
    fit_point_num = 10
    if fit_points_indices_phi2.size > fit_point_num:
        middle_index = int(fit_points_indices_phi2.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices_phi2.size / 4)
        fit_points_indices_phi2 = fit_points_indices_phi2[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]
    
    # Finding a point that has the same slope as the fitted one 
    d_dr_phi0= derivative(r, phi0_lst[1, fit_points_indices_phi0], 1)
    h0_sq = h0_sq_model(r[fit_points_indices_phi0], kappa)
    d_dr_h0_sq = derivative(r[fit_points_indices_phi0], h0_sq, 1)

    d_dr_phi2= derivative(r, phi2_lst[1, fit_points_indices_phi2], 1)
    h2_sq = h2_sq_model(r[fit_points_indices_phi2], kappa)
    d_dr_h2_sq = derivative(r[fit_points_indices_phi2], h2_sq, 1)

    connection_point_phi0 = np.array([])
    delta = 10**(-3)
    while connection_point_phi0.size == 0:
        connection_point_phi0 = r[fit_points_indices_phi0][((d_dr_phi0> d_dr_h0_sq - delta) & (d_dr_phi0< d_dr_h0_sq + delta))]
        delta += 10**(-3)
    connection_point_phi0 = connection_point_phi0[0]

    connection_point_phi2 = np.array([])
    delta = 10**(-3)
    while connection_point_phi2.size == 0:
        connection_point_phi2 = r[fit_points_indices_phi2][((d_dr_phi2> d_dr_h2_sq - delta) & (d_dr_phi2< d_dr_h2_sq + delta))]
        delta += 10**(-3)
    connection_point_phi2 = connection_point_phi2[0]
    
    # Finding the factor so that h0 fits on the curve
    a_phi0 = phi0(connection_point_phi0, NNmax, aHO) / h0_sq_model(connection_point_phi0, kappa)
    a_phi2 = phi2(connection_point_phi2, NNmax, aHO) / h2_sq_model(connection_point_phi2, kappa)

    def phi0_cont(r):
        phi0_cont = []
        for i in range(r.size):
            if r[i] < connection_point_phi0:
                phi0_cont.append(phi0_lst[1, i])
            else:
                phi0_cont.append(h0_sq_model_a(r[i], a_phi0, kappa))
        return np.array(phi0_cont)

    def phi2_cont(r):
        phi2_cont = []
        for i in range(r.size):
            if r[i] < connection_point_phi2:
                phi2_cont.append(phi2_lst[1, i])
            else:
                phi2_cont.append(h2_sq_model_a(r[i], a_phi2, kappa))
        return np.array(phi2_cont)
    

    rrms = np.sqrt(trapezoid((phi0_cont(r) + phi2_cont(r)) * r**4, x=r)/4)

    return kappa, a_phi0, connection_point_phi0, a_phi2, connection_point_phi2, rrms, fit_points_indices_phi0, fit_points_indices_phi2
        

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

aHO_lst = np.array([2.4, 2.6]) 
r = np.linspace(0, 100, 1000)


def plot_wavefunc_continuation_rrms(r, NNmax, NNmin, dNmax):
    Nmax_lst = np.arange(NNmin, NNmax + 1, dNmax)
    
    for j in range(aHO_lst.size):
        plt.figure(figsize=(17,10.6))
        # plt.subplot(121)

        rrms_lst = []
        for Nmax in Nmax_lst:
            kappa, a0, connect0, a2, connect2, rrms, fit_points0, fit_points2 = find_cont_par_cut(Nmax, aHO_lst[j])
            r_cont0 = r[r>connect0]
            r_cont2 = r[r>connect2]
            phi0_plt = plt.plot(r, phi0(r, Nmax, aHO_lst[j]))
            phi0_color = phi0_plt[0].get_color()
            plt.plot(r_cont0, h0_sq_model_a(r_cont0, a0, kappa), "--", color=phi0_color)
            phi2_plt = plt.plot(r, phi2(r, Nmax, aHO_lst[j]))
            phi2_color = phi2_plt[0].get_color()
            plt.plot(r_cont2, h2_sq_model_a(r_cont2, a2, kappa), "--", color=phi2_color)
            # plt.scatter(fit_points[0], np.exp(fit_points[1]))
            plt.semilogy()
            plt.grid(linestyle='--')
            rrms_lst.append(rrms)
        
        plt.title("Betragsquadrat der Wellenfunktionen und  Fortzetzung mit festem $\kappa = $" + str(kappa) + " 1/fm für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 20)
        plt.xlabel("r (fm)")
        plt.ylabel('$|\Phi(r)|^2$')
        legend_phi0 = np.array(["$N_{max} = $" + str(Nmax) + ", $\Phi_0(r)^2$" for Nmax in Nmax_lst])
        legend_h0 = np.array(["$N_{max} = $" + str(Nmax) + ", $\Phi_0(r)^2$-Fortsetzung" for Nmax in Nmax_lst])
        legend_phi2 = np.array(["$N_{max} = $" + str(Nmax) + ", $\Phi_2(r)^2$" for Nmax in Nmax_lst])
        legend_h2 = np.array(["$N_{max} = $" + str(Nmax) + ", $\Phi_2(r)^2$-Fortsetzung" for Nmax in Nmax_lst])
        legend = np.array([legend_phi0, legend_h0, legend_phi2, legend_h2]).transpose().flatten()
        plt.legend(legend, ncol=2)
        
        plt.savefig(f"rrms_wavefunc_continuation_wavefuncs_aHO{aHO_lst[j]}_NNmax{NNmax}_Nstep{dNmax}_kappa{kappa}.pdf", pad_inches=0.05)
        
        rmax, dr = max(r), r[1]-r[0]
        rrms_full_wave = rrms_integral_trapz(NNmax, NNmin-3, aHO_lst[j], dNmax, rmax, dr)
        # plt.subplot(122)
        plt.figure(figsize=(12,7.45))
        plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
        plt.plot(Nmax_lst, rrms_lst, 'x-')
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und Wellenfunktion mit Fortsetzung mit festem $\kappa = $' + str(kappa) + ' 1/fm')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
        plt.grid(linestyle='--')
        plt.savefig(f"rrms_wavefunc_continuation_rrms_{aHO_lst[j]}_NNmax{NNmax}_Nstep{dNmax}_kappa{kappa}.pdf")


   
        
plot_wavefunc_continuation_rrms(r, 60, 3, 5)
# plt.plot(r, h2_sq_model(r, kappa))


plt.show()