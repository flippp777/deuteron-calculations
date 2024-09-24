import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sys import exit


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

def find_cont_par_cut(NNmax, aHO, norm=True):
    # NNmax has to be bigger than 3 because Nmax=2 is the firts referece point to check for same densities
    if NNmax < 3:
        exit("NNmax muss mindestens 3 sein / NNmin darf nicht kleiner als 3 sein")

    # Define r range
    dr = 0.1
    r = np.arange(0, 100 + dr, dr)
   
    # create first list with Nmax=1 and then compute all other lists in parallel. Nmax goes from 2 to the closest number below NNmax that can be reached in steps the size of dNmax
    density_powr4 = np.array([deut_density(r, 1, aHO) * r**4, deut_density(r, NNmax-1, aHO) * r**4, deut_density(r, NNmax, aHO) * r**4])

    log_density = np.log(density_powr4)
    
    # Finding the starting point of the linear range. It is chosen as the closest point to Nmax=1 density after r=3
    r3_index = int(3/dr)
    diff_log_density_Nmax1 = np.abs(log_density[0, r3_index:] - log_density[-1, r3_index:])
    start_linear_index = max(np.where(diff_log_density_Nmax1 < min(diff_log_density_Nmax1) + 0.1)[0]) + r3_index
    start_linear_index = np.where(r ==7)[0][0]
   
    # Difference between log density at Nmax_j to Nmax_i starting from the start point of the linear range
    diff_log_density = np.abs(log_density[1, start_linear_index:] - log_density[2, start_linear_index:])
    
    # Indiexes are at the points, where the difference in log densities is smaller 0.1
    distance = 0.01
    fit_points_indices = np.where(diff_log_density < distance)[0] + start_linear_index
    
    # condition to have two differen points in order to fit a straight
    if np.unique(fit_points_indices).size < 2:
        delta = distance + 10**(-6)
        while np.unique(fit_points_indices).size < 2:
            fit_points_indices = np.append(fit_points_indices, np.where(diff_log_density < delta)[0] + start_linear_index)
            delta += 10**(-6)
    
    # Removing duplicates
    fit_points_indices = np.unique(fit_points_indices).astype(int)
    
    # use only the fit_point_num number of points in the middle of the list
    fit_point_num = 4
    if fit_points_indices.size > fit_point_num:
        middle_index = int(fit_points_indices.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices.size / 4)
        fit_points_indices = fit_points_indices[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]

    
    # fit points from fit_pint_indexes
    r_fit = r[fit_points_indices]
    log_density_fit = log_density[2, fit_points_indices]
    fit_points = np.array([r_fit, log_density_fit])
    
    # fitting the points to finde the slope
    slope = curve_fit(log_model, r_fit, log_density_fit)[0][0]
    
    # Finding a point that has the same slope as the fitted one 
    d_dr = derivative(r, log_density[2, min(fit_points_indices):max(fit_points_indices) + 1], 1)
    connection_point = np.array([])
    delta = 10**(-6)
    while connection_point.size == 0:
        connection_point = r[min(fit_points_indices):max(fit_points_indices) + 1][((d_dr > slope - delta) & (d_dr < slope + delta))]
        delta += 10**(-6)
    connection_point = connection_point[0]
    
    # Finding the y intersection
    y_intersect = np.log(deut_density(connection_point, NNmax , aHO) * connection_point**4) - slope * connection_point
    

    r_wave = r[r <= connection_point]
    r_cont = r[r > connection_point]
    
    density_powr4_lst = deut_density(r_wave, NNmax, aHO) * r_wave**4
    cont_lst = np.exp(log_model(r_cont, slope, y_intersect))
    
    if norm:    
        norm_integ_wave = np.trapz(density_powr4_lst * r_wave**(-2), x=r_wave)
        norm_integ_cont = np.trapz(cont_lst * r_cont**(-2), x=r_cont)
        
        norm_integ = norm_integ_wave + norm_integ_cont
        
        norm_factor = 1 / norm_integ

        rsq_wave = np.trapz(density_powr4_lst * norm_factor, x=r_wave) / 4
        rsq_cont = np.trapz(cont_lst * norm_factor, x=r_cont) / 4

        rrms = np.sqrt(rsq_wave + rsq_cont)

        return slope, y_intersect, connection_point, rrms, norm_factor, fit_points
        
    else:
        rsq_wave = np.trapz(density_powr4_lst, x=r_wave) / 4
        rsq_cont = np.trapz(cont_lst, x=r_cont) / 4
        
        rrms = np.sqrt(rsq_wave + rsq_cont)

        return slope, y_intersect, connection_point, rrms, fit_points
    
# print(find_cont_par_cut(10, 1.4, norm=False))