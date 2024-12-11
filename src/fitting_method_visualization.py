import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from lmfit import Model, Parameters, create_params
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sys import exit
import sys
from uncertainties import unumpy as unp

if sys.platform == 'win32':
    sys.path.insert(1, r'C:\Users\phili.DESKTOP\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc
from plotstyle.colorFader import colorFader
from plotstyle.set_size import set_size

plt.style.use('src\stylesheet.mplstyle')

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
    
    inital_fitpoint_indices = fit_points_indices_L0

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
        
    mkrit_fitpoint_indices = fit_points_indices_L0

    
    # fit points from fit_pint_indexes
    fit_points_L0 = np.array([r[fit_points_indices_L0], wavefunc_L0[1, fit_points_indices_L0]])
    fit_points_L2 = np.array([r[fit_points_indices_L2], wavefunc_L2[1, fit_points_indices_L2]])    
    
    fit_point_mean_L0 = np.mean(fit_points_L0[1])
    fit_point_std_L0 = np.std(fit_points_L0[1])
    fit_points_L0 = fit_points_L0.transpose()[(
        (fit_points_L0[1] <= fit_point_mean_L0 + 1.5 * fit_point_std_L0) & 
        (fit_points_L0[1] >= fit_point_mean_L0 - 1.5 * fit_point_std_L0))].transpose()

    fit_point_mean_L2 = np.mean(fit_points_L2[1])
    fit_point_std_L2 = np.std(fit_points_L2[1])
    fit_points_L2 = fit_points_L2.transpose()[(
        (fit_points_L2[1] <= fit_point_mean_L2 + 1.5 * fit_point_std_L2) & 
        (fit_points_L2[1] >= fit_point_mean_L2 - 1.5 * fit_point_std_L2))].transpose()
    
    final_fitpoints = fit_points_L0
    
    params = create_params(a=1, kappa=dict(value=0.2305, min=0.1, max=0.4))
    
    Model_L0 = Model(model_a_L0)

    result_L0 = Model_L0.fit(fit_points_L0[1], r=fit_points_L0[0], params=params)
    kappa_L0 = result_L0.best_values["kappa"]
    a0_fit = result_L0.best_values["a"]

    Model_L2 = Model(model_a_L2)

    result_L2 = Model_L2.fit(fit_points_L2[1], r=fit_points_L2[0], params=params)
    kappa_L2 = result_L2.best_values["kappa"]
    
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


    inital_fitpoint_indices = [element for element in inital_fitpoint_indices if element not in mkrit_fitpoint_indices]

    fig = plt.figure(figsize=set_size(502.14963))
    
    plt.plot(r, wavefunc_L0[0], color='#0000ff')
    plt.plot(r,wavefunc_L0[1], color='#ff0000')
    plt.plot(r[inital_fitpoint_indices], wavefunc_L0[1, inital_fitpoint_indices], 'd', markersize=4)
    plt.plot(r[mkrit_fitpoint_indices], wavefunc_L0[1, mkrit_fitpoint_indices], 'd', markersize=4)
    plt.plot(final_fitpoints[0], final_fitpoints[1], 'd', markersize=4)
    plt.plot([0,17], [fit_point_mean_L0, fit_point_mean_L0])
    std_plot = plt.plot([0,17], [fit_point_mean_L0 + 1.5 * fit_point_std_L0, fit_point_mean_L0 + 1.5 * fit_point_std_L0])
    plt.plot([2,17], [fit_point_mean_L0 - 1.5 * fit_point_std_L0, fit_point_mean_L0 - 1.5 * fit_point_std_L0], color=std_plot[0].get_color())
    plt.arrow(5, fit_point_mean_L0, 0, 1.5*fit_point_std_L0, length_includes_head=True, head_width=0.1, head_length=0.01, color='black')
    plt.arrow(5, fit_point_mean_L0+1.5*fit_point_std_L0, 0, -1.5*fit_point_std_L0, length_includes_head=True, head_width=0.1, head_length=0.01, color='black')
    plt.arrow(5, fit_point_mean_L0, 0, -1.5*fit_point_std_L0, length_includes_head=True, head_width=0.1, head_length=0.01, color='black')
    plt.arrow(5, fit_point_mean_L0-1.5*fit_point_std_L0, 0, +1.5*fit_point_std_L0, length_includes_head=True, head_width=0.1, head_length=0.01, color='black')
    plt.xlim(2,17)
    plt.ylim(10**(-2),1)
    plt.ylabel('[\(1/\\sqrt{\\text{fm}}\)]')
    plt.xlabel('\(r\) [fm]')
    plt.semilogy()
    plt.grid(linestyle='--')
    plt.legend(['\(|u_{\\Tilde{N}_\\text{max}}(r)|\)', '\(|u_{\\Tilde{N}_\\text{max}-1}(r)|\)', 'Initial fit points', 'Fit points after \(m\)-Criterion', 'Final fit points', 'Mean'])




    return kappa_L0, a_L0, connection_point_L0, kappa_L2, a_L2, connection_point_L2, fit_points_L0, fit_points_L2

find_continuation_parameters(30, 1.2, 'u/w')

plt.show()