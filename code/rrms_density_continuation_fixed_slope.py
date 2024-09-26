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

def exp_model(x, a, b):
    return a * np.exp(b * x)

def log_model(x, a, c):
    return a * x + c

def find_slope(aHO):
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
    
    density = np.array([deut_density(r, 1, aHO), deut_density(r, 60-1, aHO), deut_density(r, 60, aHO)])
    log_density = np.log(density)
    # Finding the starting point of the exponential range. It is chosen as the closest point to Nmax=1 density after r=3

    r3_index = int(3/dr)
    diff_log_density_Nmax1 = np.abs(log_density[0, r3_index:] - log_density[-1, r3_index:])
    start_linear_index = max(np.where(diff_log_density_Nmax1 < min(diff_log_density_Nmax1) + 0.1)[0]) + r3_index
    
    # Difference between log density at Nmax_j to Nmax_i starting from the start point of the linear range
    diff_log_density = np.abs(log_density[1, start_linear_index:] - log_density[2, start_linear_index:])

    # Indiexes are at the points, where the difference in log densities is smaller 0.1
    distance = 0.01
    fit_points_indices = np.where(diff_log_density < distance)[0] + start_linear_index
    
    # condition to have two differen points in order to fit a straight
    if np.unique(fit_points_indices).size < 10:
        delta = distance + 10**(-6)
        while np.unique(fit_points_indices).size < 10:
            fit_points_indices = np.append(fit_points_indices, np.where(diff_log_density < delta)[0] + start_linear_index)
            delta += 10**(-6)
    
    # Removing duplicates
    fit_points_indices = np.unique(fit_points_indices).astype(int)
    
    # use only the fit_point_num number of points in the middle of the list
    fit_point_num = 10
    if fit_points_indices.size > fit_point_num:
        middle_index = int(fit_points_indices.size / 2)
        two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices.size / 4)
        fit_points_indices = fit_points_indices[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]

    # fit points from fit_pint_indexes
    r_fit = r[fit_points_indices]
    log_density_fit = log_density[2, fit_points_indices]
    fit_points = np.array([r_fit, log_density_fit])
    
    # fitting the points to finde the slope
    slope = curve_fit(log_model, r_fit, log_density_fit)[0][0]

    return slope

slope_aHO12 = find_slope(1.2)
slope_aHO14 = find_slope(1.4)
slope_aHO16 = find_slope(1.6)
slope_aHO18 = find_slope(1.8)
slope_aHO20 = find_slope(2.0)
    

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
    
    # Finding the starting point of the exponential range. It is chosen as the closest point to Nmax=1 density after r=3
    r3_index = int(3/dr)
    diff_log_density_Nmax1 = np.abs(log_density[0, r3_index:] - log_density[-1, r3_index:])
    start_exp_index = max(np.where(diff_log_density_Nmax1 < min(diff_log_density_Nmax1) + 0.1)[0]) + r3_index
    
    # Difference between log density at Nmax_j to Nmax_i starting from the start point of the linear range
    diff_log_density = np.abs(log_density[1, start_exp_index:] - log_density[2, start_exp_index:])

    # Indiexes are at the points, where the difference in log densities is smaller 0.1
    distance = 0.01
    linear_range_indices = np.where(diff_log_density < distance)[0] + start_exp_index
    
    # Removing duplicates
    linear_range_indices = np.unique(linear_range_indices).astype(int)

    # condition to have two differen points in order to fit a straight
    if np.unique(linear_range_indices).size < 2:
        delta = distance + 10**(-6)
        while np.unique(linear_range_indices).size < 2:
            linear_range_indices = np.append(linear_range_indices, np.where(diff_log_density < delta)[0] + linear_range_indices)
            delta += 10**(-6)
    
    # Removing duplicates
    linear_range_indices = np.unique(linear_range_indices).astype(int)
    
    # use only the fit_point_num number of points in the middle of the list
    fit_point_num = 4
    if linear_range_indices.size > fit_point_num:
        middle_index = int(linear_range_indices.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(linear_range_indices.size / 4)
        linear_range_indices = linear_range_indices[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]
    
    if aHO == 1.2:
        slope = slope_aHO12
    elif aHO == 1.4:
        slope = slope_aHO14
    elif aHO == 1.6:
        slope = slope_aHO16
    elif aHO == 1.8:
        slope = slope_aHO18
    elif aHO == 2.0:
        slope = slope_aHO20
    else:
        exit("Wrong aHO")
    
    # Finding a point that has the same slope as the fitted one 
    d_dr = derivative(r, log_density[2, linear_range_indices], 1)
    connection_point = np.array([])
    delta = 10**(-6)
    while connection_point.size == 0:
        connection_point = r[linear_range_indices][((d_dr > slope - delta) & (d_dr < slope + delta))]
        delta += 10**(-6)
    connection_point = connection_point[0]
    
    # Finding the y intersection
    y_intersect = np.log(deut_density(connection_point, NNmax , aHO)) - slope * connection_point
    

    r_wave = r[r <= connection_point]
    r_cont = r[r > connection_point]
    
    density_lst = deut_density(r_wave, NNmax, aHO)
    cont_lst = np.exp(log_model(r_cont, slope, y_intersect))
    
    if norm:    
        norm_integ_wave = np.trapz(density_lst * r_wave**2, x=r_wave)
        norm_integ_cont = np.trapz(cont_lst * r_cont**2, x=r_cont)
        
        norm_integ = norm_integ_wave + norm_integ_cont
        
        norm_factor = 1 / norm_integ

        rsq_wave = np.trapz(density_lst * norm_factor * r_wave**4, x=r_wave) / 4
        rsq_cont = np.trapz(cont_lst * norm_factor * r_cont**4, x=r_cont) / 4

        rrms = np.sqrt(rsq_wave + rsq_cont)

        return slope, y_intersect, connection_point, rrms, norm_factor, np.array([r[linear_range_indices], density[2, linear_range_indices]])
        
    else:
        rsq_wave = np.trapz(density_lst * r_wave**4, x=r_wave) / 4
        rsq_cont = np.trapz(cont_lst * r_cont**4, x=r_cont) / 4
        
        rrms = np.sqrt(rsq_wave + rsq_cont)

        return slope, y_intersect, connection_point, rrms, np.array([r[linear_range_indices], density[2, linear_range_indices]])

aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0]) 

for aHO in aHO_lst:    
    find_cont_par_cut(60, aHO)
plt.show()