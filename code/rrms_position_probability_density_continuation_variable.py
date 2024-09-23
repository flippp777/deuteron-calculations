import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Test lists
aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0]) 
Nmax_lst = np.arange(0, 62, 2)
r = np.linspace(0, 100, 1000)

from make_position_space_wavefunc import deuteron_wavefunc
def deut_density(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[3]


def log_model(x, a, b):
    return a * x + b

def find_cont_par_cut(Nmax, aHO):
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
    
    log_density = np.log(deut_density(r, Nmax, aHO))
    d_dr = np.gradient(log_density)/dr
    d2_dr2 = np.gradient(d_dr)/dr
    d3_dr3 = np.gradient(d2_dr2)/dr
    
    d2_dr2_fit_indizes = []
    for i in range(r.size-1):
        if (d2_dr2[i] > 0 and d2_dr2[i + 1] < 0) or (d2_dr2[i] < 0 and d2_dr2[i + 1] > 0):
            d2_dr2_fit_indizes.append(i)
            
    if d2_dr2_fit_indizes == []:
        d2_dr2_fit_indizes.append(int(np.where(d2_dr2 == max(d2_dr2))[0][0]))
    
    end_linear = r[d2_dr2_fit_indizes[-1]] - 1.5
    if end_linear < 5:
        linear_range = np.arange(end_linear - 1, end_linear, 0.1)
    else:
        linear_range = np.arange(5, end_linear, 0.1)
    fit_points = np.array([linear_range, deut_density(linear_range, Nmax, aHO)]).transpose()
    fit_points = fit_points[fit_points[:, 1] > 10**(-6)]
    fit_points = fit_points.transpose()
    fit_points = np.array([fit_points[0], np.log(fit_points[1])])
    
    slope = curve_fit(log_model, fit_points[0], fit_points[1])[0][0]

    i = int(np.where(r == 5)[0][0])
    d_dr = np.array([r[i:], d_dr[i:]]).transpose()
    connection_point = np.array([])
    delta = 10**(-6)
    while connection_point.size == 0:
        connection_point = d_dr[((d_dr[:, 1] > slope - delta) & (d_dr[:, 1] < slope + delta))]
        delta += 10**(-6)

    y_intersect = np.log(deut_density(connection_point[0,0], Nmax, aHO)) - slope * connection_point[0,0]

    popt = np.array([slope, y_intersect])
    
    return popt, connection_point[0,0]

def norm_cont_density(r, Nmax, aHO):
    popt, cut = find_cont_par_cut(Nmax, aHO)
    
    if popt.size != 0:
        r_wave = r[r <= cut]
        r_cont = r[r > cut]
        
        norm_integ_wave = np.trapz(deut_density(r_wave, Nmax, aHO) * r_wave**2, x=r_wave)
        norm_integ_cont = np.trapz(np.exp(log_model(r_cont, popt[0], popt[1])) * r_cont**2, x=r_cont)
        
        norm_integ = norm_integ_wave + norm_integ_cont
        
        norm_factor = 1 / norm_integ
        
        return norm_factor
    else: 
        return np.array([])


def rrms_cont_denstity(r, Nmax, aHO):
    popt, cut = find_cont_par_cut(Nmax, aHO)
    if popt.size != 0:
        r_wave = r[r <= cut]
        r_cont = r[r > cut]
        
        density_lst = deut_density(r_wave, Nmax, aHO)
        cont_lst = np.exp(log_model(r_cont, popt[0], popt[1])) 
        
        norm_integ_wave = np.trapz(density_lst * r_wave**2, x=r_wave)
        norm_integ_cont = np.trapz(cont_lst * r_cont**2, x=r_cont)
        
        norm_integ = norm_integ_wave + norm_integ_cont
        norm_factor = 1 / norm_integ

        rsq_wave = np.trapz(density_lst * norm_factor * r_wave**4, x=r_wave) / 4
        rsq_cont = np.trapz(cont_lst * norm_factor * r_cont**4, x=r_cont) / 4
        
        rrms = np.sqrt(rsq_wave + rsq_cont)
        
        return rrms
    else: 
        return np.array([])





def rsq_integral_trapz(Nmax, aHO, rmax, dr):
    r_size = int(np.round(rmax / dr))
    r = np.linspace(0, rmax + 1, r_size)
    integr = deut_density(r, Nmax, aHO) * r**4
    return np.trapz(integr, r, dr) / 4      # Warum Faktor 4?


def rrms_integral_trapz(Nmax, aHO, Nstep, rmax, dr):
    rrms = []
    Nmax_lst = np.arange(0, Nmax + 1, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rsq_integral_trapz, Nmax, aHO, rmax, dr) for Nmax in Nmax_lst]
        for future in futures:
            rrms.append(future.result())
    
    rrms = np.sqrt(np.array(rrms))
    
    return np.array([Nmax_lst, rrms])


def test_plots(r, NNmax, dNmax):
    Nmax_lst = np.arange(1, NNmax + dNmax, dNmax)
    
    for j in range(aHO_lst.size):
        rrms = []
        plt.figure()
        plt.subplot(121)
        for Nmax in Nmax_lst:
            popt , cut = find_cont_par_cut(Nmax, aHO_lst[j])
            r_cont = np.array([cut, 20])
            norm = norm_cont_density(r, Nmax, aHO_lst[j])
            
            plt.plot(r, deut_density(r, Nmax, aHO_lst[j]) * norm, 'b')
            plt.plot(r_cont, np.exp(log_model(r_cont, popt[0], popt[1])) * norm, "r")
            plt.semilogy()
            plt.grid(linestyle='--')
            
            rrms.append(rrms_cont_denstity(r, Nmax, aHO_lst[j]))
            
        plt.title("Wellenfunktionen und exponentielle Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 20)
        plt.legend(["Wellenfunktion", "Fortsetzung"])
        
        rmax, dr = max(r), r[1]-r[0]
        rrms_full_wave = rrms_integral_trapz(60, 1.4, 2, rmax, dr)
        plt.subplot(122)
        plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
        plt.plot(Nmax_lst, rrms, 'x-')
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und exponentieller Fortsetzung')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
        plt.grid(linestyle='--')


def test_d(r, NNmax, dNmax):
    Nmax_lst = np.arange(1, NNmax + dNmax, dNmax)
    
    for j in range(aHO_lst.size):
        plt.figure()
        plt.subplot(221)
        legend = []
        for Nmax in Nmax_lst:
            plt.plot(r, np.log(deut_density(r, Nmax, aHO_lst[j])))
            legend.append(str(Nmax))
        plt.legend(legend)
        plt.title('aHO = ' + str(aHO_lst[j]))
        plt.grid(linestyle='--')
        plt.subplot(222)
        legend = []
        for Nmax in Nmax_lst:
            plt.plot(r, np.gradient(np.log(deut_density(r, Nmax, aHO_lst[j])))/0.1)
            legend.append(str(Nmax))
        plt.legend(legend)
        plt.title('aHO = ' + str(aHO_lst[j]) + ', d/dr')
        plt.grid(linestyle='--')
        plt.subplot(223)
        legend = []
        for Nmax in Nmax_lst:
            plt.plot(r, np.gradient(np.gradient(np.log(deut_density(r, Nmax, aHO_lst[j])))))
            legend.append(str(Nmax))
        plt.legend(legend)
        plt.title('aHO = ' + str(aHO_lst[j]) + ', d2/dr2')
        plt.grid(linestyle='--')
        plt.subplot(224)
        legend = []
        for Nmax in Nmax_lst:
            plt.plot(r, np.gradient(np.gradient(np.gradient(np.log(deut_density(r, Nmax, aHO_lst[j]))))))
            legend.append(str(Nmax))
        plt.legend(legend)
        plt.title('aHO = ' + str(aHO_lst[j])+ ', d3/dr3')
        plt.grid(linestyle='--')
            
test_plots(r, 10, 2)

# test_d(np.arange(0,15,0.1), 5, 1)

plt.show()