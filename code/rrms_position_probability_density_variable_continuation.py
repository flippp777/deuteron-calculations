import numpy as np
import matplotlib.pyplot as plt

from make_position_space_wavefunc import deuteron_wavefunc

r_model = np.linspace(0, 15, 500)

def deut_density(Nmax, aHO, r):
    return deuteron_wavefunc(Nmax, aHO, r)[3]

def find_fit_points(Nmax, aHO, delta_d2_dr2, r):
    log_density = np.log(deut_density(Nmax, aHO, r))
    d_dr = np.gradient(log_density)
    d2_dr2 = np.gradient(d_dr)

    plot_points = []
    for i in range(log_density.size):
        if d2_dr2[i] < delta_d2_dr2 and -delta_d2_dr2 < d2_dr2[i] and d_dr[i] < 0:
            plot_points.append([r_model[i], log_density[i]])
    
    return np.array(plot_points)

fit_points = find_fit_points(20, 1.8, 10**(-5), r_model)

plt.plot(r_model, np.log(deut_density(20, 1.8, r_model)))
plt.scatter(fit_points[:, 0], fit_points[:, 1])

plt.show()