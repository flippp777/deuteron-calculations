import numpy as np
from scipy.optimize import curve_fit

from make_position_space_wavefunc import deuteron_wavefunc


def deut_density(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[3]

def find_fit_points_and_cut(r, Nmax, aHO, delta_d2_dr2):
    log_density = np.log(deut_density(r, Nmax, aHO))
    d_dr = np.gradient(log_density)
    d2_dr2 = np.gradient(d_dr)

    plot_points = []
    for i in range(log_density.size):
        if d2_dr2[i] < delta_d2_dr2 and -delta_d2_dr2 < d2_dr2[i] and d_dr[i] < 0:
            plot_points.append([r[i], log_density[i]])
    
    plot_points = np.array(plot_points)
    cut = (max(plot_points[:, 0]) - min(plot_points[:, 0])) / 2
    
    return plot_points, cut

def log_model(x, a, b):
    return a * x + b

def log_cont_par_and_cut(r, Nmax, aHO, delta_d2_dr2):
    fit_points, cut = find_fit_points_and_cut(r, Nmax, aHO, delta_d2_dr2)
    
    popt = curve_fit(log_model, fit_points[:, 0], fit_points[:, 1])[0]
    
    return popt, cut

def norm_cont_density(r, Nmax, aHO, delta_d2_dr2):
    popt, cut = log_cont_par_and_cut(r, Nmax, aHO, delta_d2_dr2)
    
    r_wave = r[r <= cut]
    r_cont = r[r > cut]
    
    norm_integ_wave = np.trapz(deut_density(r_wave, Nmax, aHO) * r_wave**2, r_wave)
    norm_integ_cont = np.trapz(np.exp(log_model(r_cont, popt[0], popt[1])) * r_cont**2, r_cont)
    
    norm_integ = norm_integ_wave + norm_integ_cont
    
    norm_factor = 1 / norm_integ
    
    return norm_factor

def rrms_and_cut_cont_denstity(r, Nmax, aHO, delta_d2_dr2):
    popt, cut = log_cont_par_and_cut(r, Nmax, aHO, delta_d2_dr2)
    
    r_wave = r[r <= cut]
    r_cont = r[r > cut]
    
    density_lst = deut_density(r_wave, Nmax, aHO)
    cont_lst = np.exp(log_model(r_cont, popt[0], popt[1])) 
    
    norm_integ_wave = np.trapz(density_lst * r_wave**2, r_wave)
    norm_integ_cont = np.trapz(cont_lst * r_cont**2, r_cont)
    
    norm_integ = norm_integ_wave + norm_integ_cont
    norm_factor = 1 / norm_integ

    rsq_wave = np.trapz(density_lst * norm_factor, r_wave**4, r_wave)
    rsq_cont = np.trapz(cont_lst * norm_factor, r_cont**4, r_cont)
    
    rrms = np.sqrt(rsq_wave + rsq_cont)
    
    return rrms, cut