import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from lmfit import Model, Parameters, create_params
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sys import exit
import sys

if sys.platform == 'win32':
    sys.path.insert(1, r'C:\Users\phili.DESKTOP\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc

def deut_density(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[3]

# Phi functions

def phi0(r, Nmax, aHO):
    return np.abs(deuteron_wavefunc(Nmax, aHO, r)[1])

def phi2(r, Nmax, aHO):
    return np.abs(deuteron_wavefunc(Nmax, aHO, r)[2])

def h0_model(r, kappa):
    return 1 / (kappa * r) * np.exp(-kappa * r)

def h0_model_a(r, a, kappa):
    return a / (kappa * r) * np.exp(-kappa * r)

def h2_model(r, kappa):
    return 1/(kappa * r) * np.exp(-kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2)

def h2_model_a(r, a, kappa):
    return a/(kappa * r) * np.exp(-kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2)

# u/w functions

def u(r, Nmax, aHO):
    return np.abs(r * deuteron_wavefunc(Nmax, aHO, r)[1])

def w(r, Nmax, aHO):
    return np.abs(r * deuteron_wavefunc(Nmax, aHO, r)[2])

def rh0_model(r, kappa):
    return r/ (kappa * r) * np.exp(-kappa * r)

def rh0_model_a(r, a, kappa):
    return a * r / (kappa * r) * np.exp(-kappa * r)

def rh2_model(r, kappa):
    return r/(kappa * r) * np.exp(-kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2)

def rh2_model_a(r, a, kappa):
    return a * r/(kappa * r) * np.exp(-kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2)



def find_continuation_parameters(NNmax, aHO, wavefunc, number_at_least_collected_points=10, number_used_points=50, norm=True):
    # NNmax has to be bigger than 3 because Nmax=2 is the firts referece point to check for same densities
    if NNmax < 1:
        exit("NNmax muss mindestens 1 sein / NNmin darf nicht kleiner als 1 sein")

    # Define r range
    dr = 0.1
    r = np.arange(0, 150 + dr, dr)
   
    # create first list with Nmax=1 and then compute all other lists in parallel. Nmax goes from 2 to the closest number below NNmax that can be reached in steps the size of dNmax
    if wavefunc == 'phi':
        wavefunc_L0 = np.array([phi0(r, NNmax-1, aHO), phi0(r, NNmax, aHO)])
        wavefunc_L2 = np.array([phi2(r, NNmax-1, aHO), phi2(r, NNmax, aHO)])

        def model_a_L0(r, a, kappa):
            return h0_model_a(r, a, kappa)
        def model_a_L2(r, a, kappa):
            return h2_model_a(r, a, kappa)

        def model_L0(r, kappa):
            return h0_model(r, kappa)
        def model_L2(r, kappa):
            return h2_model(r, kappa)

        def L0(r, Nmax, aHO):
            return phi0(r, Nmax, aHO)
        def L2(r, Nmax, aHO):
            return phi2(r, Nmax, aHO)
    elif wavefunc == 'u/w':
        wavefunc_L0 = np.array([u(r, NNmax-1, aHO), u(r, NNmax, aHO)])
        wavefunc_L2 = np.array([w(r, NNmax-1, aHO), w(r, NNmax, aHO)])

        def model_a_L0(r, a, kappa):
            return rh0_model_a(r, a, kappa)
        def model_a_L2(r, a, kappa):
            return rh2_model_a(r, a, kappa)

        def model_L0(r, kappa):
            return rh0_model(r, kappa)
        def model_L2(r, kappa):
            return rh2_model(r, kappa)

        def L0(r, Nmax, aHO):
            return u(r, Nmax, aHO)
        def L2(r, Nmax, aHO):
            return w(r, Nmax, aHO)
    else:
        sys.exit('wrong wavefunc parametrization')

    log_wavefunc_L0 = np.log(wavefunc_L0)
    log_wavefunc_L2 = np.log(wavefunc_L2)
    
    
    # Finding the starting point of the fit range. Minimum in range (0, 4)
    # ddr_log_phi0 = derivative(r, log_phi0[1], 1)
    # r4_index = int(4/dr)
    # start_fit_index_phi0 = np.where(ddr_log_phi0[:r4_index] == min(ddr_log_phi0[:r4_index]))[0][0]
    start_index_L0 = int(3/dr)
    
    # ddr_log_phi2 = derivative(r, log_phi2[1], 1)
    # r4_index = int(4/dr)
    # start_fit_index_phi2 = np.where(ddr_log_phi2[:r4_index] == min(ddr_log_phi2[:r4_index]))[0][0]
    start_index_L2 = int(3/dr)
   
    # Difference between log density at Nmax_j to Nmax_i starting from the start point of the linear range
    diff_log_L0 = np.abs(log_wavefunc_L0[0, start_index_L0:] - log_wavefunc_L0[1, start_index_L0:])
    diff_log_L2 = np.abs(log_wavefunc_L2[0, start_index_L2:] - log_wavefunc_L2[1, start_index_L2:])
    
    # Indiexes are at the points, where the difference in log densities is smaller 0.1
    distance = 0.01
    fit_points_indices_L0 = np.where(diff_log_L0 < distance)[0] + start_index_L0
    fit_points_indices_L2 = np.where(diff_log_L2 < distance)[0] + start_index_L2
    
    # condition to have two differen points in order to fit something
    if np.unique(fit_points_indices_L0).size < number_at_least_collected_points:
        delta = distance + 10**(-3)
        while np.unique(fit_points_indices_L0).size < number_at_least_collected_points:
            fit_points_indices_L0 = np.append(fit_points_indices_L0, np.where(diff_log_L0 < delta)[0] + start_index_L0)
            delta += 10**(-3)
    if np.unique(fit_points_indices_L2).size < number_at_least_collected_points:
        delta = distance + 10**(-3)
        while np.unique(fit_points_indices_L2).size < number_at_least_collected_points:
            fit_points_indices_L2 = np.append(fit_points_indices_L2, np.where(diff_log_L2 < delta)[0] + start_index_L2)
            delta += 10**(-3)
    
    # Removing duplicates
    fit_points_indices_L0 = np.unique(fit_points_indices_L0).astype(int)
    fit_points_indices_L2 = np.unique(fit_points_indices_L2).astype(int)

    # use only the fit_point_num number of points in the middle of the list
    if fit_points_indices_L0.size > number_used_points:
        middle_index = int(fit_points_indices_L0.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices_L0.size / 4)
        fit_points_indices_L0 = fit_points_indices_L0[three_quater_index - int(number_used_points / 2):three_quater_index + int(number_used_points / 2)]
    
    if fit_points_indices_L2.size > number_used_points:
        middle_index = int(fit_points_indices_L2.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices_L2.size / 4)
        fit_points_indices_L2 = fit_points_indices_L2[three_quater_index - int(number_used_points / 2):three_quater_index + int(number_used_points / 2)]

    
    # fit points from fit_pint_indexes
    fit_points_L0 = np.array([r[fit_points_indices_L0], wavefunc_L0[1, fit_points_indices_L0]])
    fit_points_L2 = np.array([r[fit_points_indices_L2], wavefunc_L2[1, fit_points_indices_L2]])    
    # plt.figure()
    # plt.scatter(fit_points_phi2[0], fit_points_phi2[1])
    # plt.plot(r, phi2_lst[1])
    # plt.ylim(bottom=10**(-6))
    # plt.xlim(0,12)
    # plt.semilogy()
    
    params = create_params(a=1, kappa=dict(value=0.2305, min=0.1, max=0.4))
    
    Model_L0 = Model(model_a_L0)

    result_L0 = Model_L0.fit(fit_points_L0[1], r=fit_points_L0[0], params=params)
    kappa_L0 = result_L0.best_values["kappa"]

    # a0_fit = result_L0.best_values["a"]
    # plt.figure()
    # h0result.plot_fit()
    
    # plt.plot(r, h0_model_a(r, h0result.best_values['a'], kappa_phi0))
    # plt.scatter(fit_points_phi0[0], fit_points_phi0[1])
    # plt.ylim(min(fit_points_phi0[1])-1, max(fit_points_phi0[1])+1)
    # plt.xlim(min(fit_points_phi0[0])-1, max(fit_points_phi0[0])+1)
    # plt.show()

    Model_L2 = Model(model_a_L2)

    result_L2 = Model_L2.fit(fit_points_L2[1], r=fit_points_L2[0], params=params)
    kappa_L2 = result_L2.best_values["kappa"]

    # a2_fit = result_L2.best_values["a"]
    # plt.figure()
    # h2result.plot_fit()
    # plt.figure()
    # plt.plot(r, h2_model_a(r, 0.1*h0result.best_values['a'], 0.2305))
    # plt.plot(r, h2_model_a(r, h0result.best_values['a'], kappa_phi2))
    # plt.ylim(0, 0.02)
    # plt.xlim(min(fit_points_phi2[0]), max(fit_points_phi2[0]))
    # plt.show()
    
    # plt.plot(r, h2_model_a(r, h2result.best_values['a'], kappa_phi2))
    # plt.plot(r, phi2_lst[1])
    # plt.scatter(fit_points_phi2[0], fit_points_phi2[1])
    # plt.legend(['Fit', 'phi2', 'fit points'])
    # plt.semilogy()
    # plt.ylim(10**(-6), 10)
    # plt.xlim(0,20)

    # plt.figure()
    # plt.plot(r, np.abs(np.gradient(np.log(h2_model(r, kappa_phi2)), r)))
    # plt.plot(r, np.abs(np.gradient(np.log(phi2_lst[1]), r)))
    # # plt.scatter(fit_points_phi2[0], fit_points_phi2[1])
    # # plt.legend(['Fit', 'phi2', 'fit points'])
    # # plt.semilogy()
    # # plt.ylim(10**(-6), 10)
    # # plt.xlim(0,20)

    # plt.figure()
    # plt.plot(r, np.gradient(h2_model_a(r, h2result.best_values['a'], kappa_phi2), r))
    # plt.plot(r, np.gradient(phi2_lst[1], r))
    # plt.legend(['Fit', 'phi2'])
    # plt.semilogy()
    # plt.ylim(10**(-10), 10)
    # plt.xlim(0,20)

    # plt.show()
    
    
    # sys.exit()
    
    # Finding a point that has the same slope as the fitted one (up to a factor)
    d_dr_L0 = np.gradient(np.log(fit_points_L0[1]), fit_points_L0[0])[1:-1]
    L0_model = np.log(model_L0(fit_points_L0[0], kappa_L0))
    d_dr_L0_model = np.gradient(L0_model, fit_points_L0[0])[1:-1]
    connection_point_L0 = np.array([])
    delta = 10**(-3)
    while connection_point_L0.size == 0:
        connection_point_L0 = fit_points_L0[0][1:-1][((d_dr_L0 > d_dr_L0_model - delta) & (d_dr_L0 < d_dr_L0_model + delta))]
        # print(f'phi0: Nmax={NNmax}, delta={delta}')
        delta += 10**(-3)
    connection_point_L0 = connection_point_L0[0]
    
    d_dr_L2 = np.gradient(np.log(fit_points_L2[1]), fit_points_L2[0])[1:-1]
    L2_model = np.log(model_L2(fit_points_L2[0], kappa_L2))
    d_dr_L2 = np.gradient(L2_model, fit_points_L2[0])[1:-1]
    connection_point_L2 = np.array([])
    delta = 10**(-3)
    while connection_point_L2.size == 0:
        connection_point_L2 = fit_points_L2[0][1:-1][((d_dr_L2 > d_dr_L2 - delta) & (d_dr_L2 < d_dr_L2 + delta))]
        # print(f'phi2: Nmax={NNmax}, delta={delta}')
        delta += 10**(-3)
    connection_point_L2 = connection_point_L2[0]

    a_L0 = L0(connection_point_L0, NNmax, aHO) / model_L0(connection_point_L0, kappa_L0)
    a_L2 = L2(connection_point_L2, NNmax, aHO) / model_L2(connection_point_L2, kappa_L2)

    def continuation_L0(r):
        cont = []
        for i in range(r.size):
            if r[i] <= connection_point_L0:
                cont.append(wavefunc_L0[1,i])
            else:
                cont.append(model_a_L0(r[i], a_L0, kappa_L0))
        return np.array(cont)

    def continuation_L2(r):
        cont = []
        for i in range(r.size):
            if r[i] <= connection_point_L2:
                cont.append(wavefunc_L2[1,i])
            else:
                cont.append(model_a_L2(r[i], a_L2, kappa_L2))
        return np.array(cont)
    
    if wavefunc == 'phi':
        rrms = np.sqrt(trapezoid((continuation_L0(r)**2 + continuation_L2(r)**2) * r**4, x=r)/4)
    elif wavefunc == 'u/w':
        rrms = np.sqrt(trapezoid((continuation_L0(r)**2 + continuation_L2(r)**2) * r**2, x=r)/4)

    return kappa_L0, a_L0, connection_point_L0, kappa_L2, a_L2, connection_point_L2, rrms, fit_points_L0, fit_points_L2

# find_continuation_parameters(4, 1.2)
    
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

r = np.linspace(0, 100, 1000)



def plot_density_continuation_rrms(r, aHO, NNmax, NNmin, dNmax, wavefunc, number_at_least_collected_points, number_used_points, save=False):
    Nmax_lst = np.arange(NNmin, NNmax + 1, dNmax)

    if wavefunc == 'phi':
        def model_a_L0(r, a, kappa):
           return h0_model_a(r, a, kappa) 
        def model_a_L2(r, a, kappa):
           return h2_model_a(r, a, kappa) 

        def L0(r, Nmax, aHO):
            return phi0(r, Nmax, aHO)
        def L2(r, Nmax, aHO):
            return phi2(r, Nmax, aHO)
    elif wavefunc == 'u/w':
        def model_a_L0(r, a, kappa):
           return rh0_model_a(r, a, kappa) 
        def model_a_L2(r, a, kappa):
           return rh2_model_a(r, a, kappa) 

        def L0(r, Nmax, aHO):
            return u(r, Nmax, aHO)
        def L2(r, Nmax, aHO):
            return w(r, Nmax, aHO)
    
    plt.figure(figsize=(17,10.6))
    # plt.subplot(121)

    rrms_lst = []
    kappa_phi0_lst = []
    kappa_phi2_lst = []
    counter = 0
    for Nmax in Nmax_lst:
        counter += 1
        kappa_phi0, c_phi0, connect_phi0, kappa_phi2, c_phi2, connect_phi2, rrms, fit_points_phi0, fit_points_phi2 = find_continuation_parameters(Nmax, aHO, wavefunc, number_at_least_collected_points, number_used_points)

        r_cont_phi0 = np.arange(connect_phi0, 30.1, 0.1)
        r_cont_phi2 = np.arange(connect_phi2, 30.1, 0.1)

        phi0_plot = plt.plot(r, L0(r, Nmax, aHO))
        phi0_color = phi0_plot[0].get_color()
        plt.plot(r_cont_phi0, model_a_L0(r_cont_phi0, c_phi0, kappa_phi0), '--', color=phi0_color)
        # phi2_plot = plt.plot(r, L2(r, Nmax, aHO))
        # phi2_color = phi2_plot[0].get_color()
        # plt.plot(r_cont_phi2, model_a_L2(r_cont_phi2, c_phi2, kappa_phi2), "--", color=phi2_color)
        # plt.scatter(fit_points_phi0[0], fit_points_phi0[1], color=phi0_color)
        # plt.scatter(fit_points_phi2[0], fit_points_phi2[1], color=phi2_color)
        plt.semilogy()
        plt.grid(linestyle='--')
        rrms_lst.append(rrms)
        kappa_phi0_lst.append(round(kappa_phi0,4))
        kappa_phi2_lst.append(round(kappa_phi2,4))
        print(f"Nmax = {Nmax} ({100*counter/len(Nmax_lst)} %)")
        print(f'kappa_phi0 = {kappa_phi0}, kappa_phi2 = {kappa_phi2}')
        
    # np.savetxt(f'phi_fitted_kappas_aHO{aHO}.txt', np.array([Nmax_lst, kappa_phi0_lst, kappa_phi2_lst]).transpose())

    plt.title("Wellenfunktionen und angepasster Fortzetzung für aHO = " + str(aHO) + " fm (logaryhtmische y-Axhse)") 
    plt.ylim(10**(-6), 1)
    plt.xlim(0, 30)
    plt.xlabel('r (fm)')
    plt.ylabel('$|\Phi(r)|$')
    legend = ['$\Phi_0(r)$', '$\Phi_0(r)$-Fortsetzung', '$\Phi_2(r)$', '$\Phi_2(r)$-Fortsetzung']
    plt.legend(legend, ncol=2)
    # plt.legend(np.array([[f"Nmax={Nmax_lst[i]}, $\Phi_0(r)$", f"Nmax={Nmax_lst[i]}, $\Phi_0(r)$-Fortsetzung, $\kappa$={kappa_phi0_lst[i]}", f"Nmax={Nmax_lst[i]}, $\Phi_0(r)$-Fitpunkte", 
    #                       f"Nmax={Nmax_lst[i]}, $\Phi_2(r)$", f"Nmax={Nmax_lst[i]}, $\Phi_2(r)$-Fortsetzung, $\kappa$={kappa_phi2_lst[i]}", f"Nmax={Nmax_lst[i]}, $\Phi_2(r)$-Fitpunkte"] for i in range(0,Nmax_lst.size)]).flatten())
    
    if save:
        plt.savefig(f"rrms_wavefunc_continuation_wavefuncs_aHO{aHO}_NNmax{NNmax}_Nstep{dNmax}_fitted_kappa.pdf", pad_inches=0.01)
    
    rmax, dr = max(r), r[1]-r[0]
    rrms_full_wave = rrms_integral_trapz(NNmax, NNmin-1, aHO, dNmax, rmax, dr)
    # plt.subplot(122)
    plt.figure(figsize=(12,7.45))
    plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
    plt.plot(Nmax_lst, rrms_lst, 'x-')
    plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO) + ' fm aus Wellenfunktion und angepasster Fortsetzung')
    plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
    plt.xlabel('$N_{max}$')
    plt.ylabel("$R_{rms}$ (fm)")
    plt.yticks(np.append([1.2, 1.4, 1.6, 1.8, 1.96998], np.arange(2, max(rrms_lst)+0.2, 0.2)))
    # plt.yticks([1.75, 1.8, 1.85, 1.9, 1.95, 1.97, 2.0, 2.05, 2.1])
    plt.grid(linestyle='--')
    
    if save:
        plt.savefig(f"rrms_wavefunc_continuation_rrms_{aHO}_NNmax{NNmax}_Nstep{dNmax}_" + wavefunc + "_fitted_kappa.pdf")
            
def ANN_comparison(save=False):
    Nmax_lst = np.arange(1, 11)
    aHO_lst = [1.1, 1.2, 1.3]
    NNmax_lst = np.arange(4, 11)
    
    rrms_lsts = []
    rrms_fullwav = []
    for aHO in aHO_lst:
        rrms_lst = []
        for Nmax in Nmax_lst:
            rrms_lst.append(find_continuation_parameters(Nmax, aHO)[6])
        rrms_lsts.append(rrms_lst)
        rrms_fullwav.append(rrms_integral_trapz(10, 1, aHO, 1, 150, 0.01))
        
    rrms_lsts = np.array(rrms_lsts)

    mean_lst = []
    std_lst = []
    for i in NNmax_lst:
        mean_lst.append(np.mean(rrms_lsts[:, i-4:i].flatten()))
        std_lst.append(np.std(rrms_lsts[:, i-4:i].flatten()))
    
    fig = plt.figure(figsize=(12,7.45))
    gs = fig.add_gridspec(1, 2, wspace=0)
    axs = gs.subplots(sharey=True)
    
    
    aHO11_plt = axs[0].plot(Nmax_lst, rrms_lsts[0], 'o-')
    aHO11_color = aHO11_plt[0].get_color()
    axs[0].plot(rrms_fullwav[0][0], rrms_fullwav[0][1], 'o--', color = aHO11_color)
    aHO12_plt = axs[0].plot(Nmax_lst, rrms_lsts[1], 'v-')
    aHO12_color = aHO12_plt[0].get_color()
    axs[0].plot(rrms_fullwav[1][0], rrms_fullwav[1][1], 'v--', color= aHO12_color)
    aHO13_plt = axs[0].plot(Nmax_lst, rrms_lsts[2], '^-')
    aHO13_color = aHO13_plt[0].get_color()
    axs[0].plot(rrms_fullwav[2][0], rrms_fullwav[2][1], '^--', color=aHO13_color)
    axs[0].set_xlabel('$N_{max}$')
    axs[0].set_ylabel('$R_{rms}$ (fm)')
    plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2, 2.4, 2.6, 2.8, 3])
    axs[0].legend(['$a_{HO}=1.1$ fm, Fortsetzung', '$a_{HO}=1.1$ fm, Wellenfunktion',
                '$a_{HO}=1.2$ fm, Fortsetzung', '$a_{HO}=1.2$ fm, Wellenfunktion',
                '$a_{HO}=1.3$ fm, Fortsetzung', '$a_{HO}=1.3$ fm, Wellenfunktion'], loc='lower right')
    axs[0].grid(linestyle='--')
    
    ANN_lst = [1.766, 1.923, 1.927, 1.868, 1.895, 1.881, 1.917]
    ANN_err_lst = [[0.051, 0.051], [0.051, 0.051], [0.047, 0.047], [0.032, 0.032], [0.019, 0.019], [0.013, 0.017], [0.013, 0.013]]
    ANN_err_lst = [[0.051, 0.051, 0.047, 0.032, 0.019, 0.013, 0.013], [0.051, 0.051, 0.047, 0.032, 0.019, 0.017, 0.013]]
    
    
    axs[1].errorbar(NNmax_lst, mean_lst, yerr=std_lst, fmt='o', capsize=5)
    axs[1].errorbar(NNmax_lst, ANN_lst, yerr=ANN_err_lst, fmt='v', capsize=5)
    axs[1].set_xlabel('$NN_{max}$')
    # plt.ylabel('$R_{rms}$ (fm)')
    # axs[1].yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
    axs[1].legend(['Mittelwert der Fortsetzungen', 'ANN Vorgersage'], loc='lower right')
    axs[1].grid(linestyle='--')
    
    if save:
        plt.savefig("ANN_comparison_fitted_kappa.pdf", pad_inches=0)
    
# ANN_comparison(True)

plot_density_continuation_rrms(r, 2, 60, 1, 1, 'u/w', 10, 50, False)
# plot_density_continuation_rrms(r, 1.2, 60, 1, 1, 'u/w', 10, 50, True)
# print(find_continuation_parameters(150, 1.4, 'u/w', 10, 50))

# aHO_lst = np.array([2.0]) 
        
# plot_density_continuation_rrms(r, 60, 3, 1, 30, 15, True)


plt.show()