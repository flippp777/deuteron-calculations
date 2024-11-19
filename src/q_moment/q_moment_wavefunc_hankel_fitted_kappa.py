import scipy as sp
import numpy as np
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import curve_fit
from lmfit import Model, Parameters, create_params

if sys.platform == "win32":
    sys.path.insert(1, r'C:\Users\phili.DESKTOP\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc

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

def q(r, Nmax, aHO):
    return 1/np.sqrt(50) * r**2 * u(r, Nmax, aHO) * w(r, Nmax, aHO) - 1/20 * r**2 * w(r, Nmax, aHO)**2

def Q_trapz(r, Nmax, aHO):
    return sp.integrate.trapezoid(q(r, Nmax, aHO), x=r)


def continue_q_wavefunc(Nmax, aHO, number_at_least_collected_points=10, number_used_points=50):
    dr = 0.1
    r = np.arange(0, 150 + dr, dr)
    
    u_lst = np.array([u(r, Nmax-1, aHO), u(r, Nmax, aHO)])
    w_lst = np.array([w(r, Nmax-1, aHO), w(r, Nmax, aHO)])
    
    log_u = np.log(u_lst)
    log_w = np.log(w_lst)
    
    # start_index_u = np.where(log_u[1] == max(log_u[1]))[0][0]
    start_index_u = int(3/dr)
    # start_index_w = np.where(log_w[1] == max(log_w[1]))[0][0]
    start_index_w = int(3/dr)

    diff_log_u = np.abs(log_u[0, start_index_u:] - log_u[1, start_index_u:])
    diff_log_w = np.abs(log_w[0, start_index_w:] - log_w[1, start_index_w:])
    
    distance = 0.01
    fit_points_indices_u = np.where(diff_log_u < distance)[0] + start_index_u
    fit_points_indices_w = np.where(diff_log_w < distance)[0] + start_index_w

    if np.unique(fit_points_indices_u).size < number_at_least_collected_points:
        while np.unique(fit_points_indices_u).size < number_at_least_collected_points:
            distance += 10**(-3)
            fit_points_indices_u = np.append(fit_points_indices_u, np.where(diff_log_u < distance)[0] + start_index_u).flatten()

    distance = 0.01
    if np.unique(fit_points_indices_w).size < number_at_least_collected_points:
        while np.unique(fit_points_indices_w).size < number_at_least_collected_points:
            distance += 10**(-3)
            fit_points_indices_w = np.append(fit_points_indices_w,np.where(diff_log_w < distance)[0] + start_index_w).flatten()

    fit_points_indices_u = np.unique(fit_points_indices_u)
    fit_points_indices_w = np.unique(fit_points_indices_w)
    
    if fit_points_indices_u.size > number_used_points:
        middle_index = int(fit_points_indices_u.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices_u.size / 4)
        fit_points_indices_u = fit_points_indices_u[three_quater_index - int(number_used_points / 2):three_quater_index + int(number_used_points / 2)]
    if fit_points_indices_w.size > number_used_points:
        middle_index = int(fit_points_indices_u.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices_w.size / 4)
        fit_points_indices_w = fit_points_indices_w[three_quater_index - int(number_used_points / 2):three_quater_index + int(number_used_points / 2)]
    
    # finding the slope with the mean of the derivative
    fit_points_u = np.array([r[fit_points_indices_u], u_lst[1, fit_points_indices_u]])
    fit_points_w = np.array([r[fit_points_indices_w], w_lst[1, fit_points_indices_w]])

    # Fit
    params_u = create_params(a=1, kappa=dict(value=0.2305))
    
    h0model = Model(rh0_model_a)
    h0result = h0model.fit(fit_points_u[1], r=fit_points_u[0], params=params_u)
    kappa_u = h0result.best_values["kappa"]
    
    params_w = create_params(a=0.01, kappa=dict(value=0.2305))
    
    h2model = Model(rh2_model_a)
    h2result = h2model.fit(fit_points_w[1], r=fit_points_w[0], params=params_w)
    kappa_w = h2result.best_values["kappa"]

    # plt.figure()
    # plt.plot(r, np.gradient(np.log(w_lst[1]), r))
    # plt.plot(r, np.gradient(np.log(rh2_model(r, kappa_w)), r))
    # plt.scatter(fit_points_w[0], fit_points_w[1])
    # plt.legend(['u', 'rh0_model'])
    # plt.xlim(0, 20)
    # plt.ylim(-10,10)
    
    # plt.show()
    
    # sys.exit()

    # Finding a point that has the same slope as the found one (the gradient of the log is used to get rid of the "a" dependence)
    d_dr_u = np.gradient(np.log(fit_points_u[1]), fit_points_u[0])[1:-1]
    rh0 = rh0_model(r[fit_points_indices_u], kappa_u)
    d_dr_rh0 = np.gradient(np.log(rh0), fit_points_u[0])[1:-1]
    connection_point_u = np.array([])
    delta = 10**(-3)
    while connection_point_u.size == 0:
        connection_point_u = r[fit_points_indices_u[1:-1]][((d_dr_u > d_dr_rh0 - delta) & (d_dr_u < d_dr_rh0 + delta))]
        delta += 10**(-3)
    connection_point_u = connection_point_u[0]
    
    d_dr_w = np.gradient(np.log(fit_points_w[1]), fit_points_w[0])[1:-1]
    rh2 = rh2_model(r[fit_points_indices_w], kappa_w)
    d_dr_rh2 = np.gradient(np.log(rh2), fit_points_w[0])[1:-1]
    connection_point_w = np.array([])
    delta = 10**(-3)
    while connection_point_w.size == 0:
        connection_point_w = r[fit_points_indices_w[1:-1]][((d_dr_w > d_dr_rh2 - delta) & (d_dr_w < d_dr_rh2 + delta))]
        delta += 10**(-3)
    connection_point_w = connection_point_w[0]

    # Finding the y intersection
    a_u = u(connection_point_u, Nmax, aHO) / rh0_model(connection_point_u, kappa_u) 
    a_w = w(connection_point_w, Nmax, aHO) / rh2_model(connection_point_w, kappa_w)
    
    def u_cont(r):
        u_cont = []
        for i in range(r.size):
            if r[i] < connection_point_u:
                u_cont.append(u_lst[1, i])
            else:
                u_cont.append(rh0_model_a(r[i], a_u, kappa_u))
        return np.array(u_cont)

    def w_cont(r):
        w_cont = []
        for i in range(r.size):
            if r[i] < connection_point_w:
                w_cont.append(w_lst[1, i])
            else:
                w_cont.append(rh2_model_a(r[i], a_w, kappa_w))
        return np.array(w_cont)

    Q = 1/np.sqrt(50) * sp.integrate.trapezoid(r**2 * u_cont(r) * w_cont(r), x=r) - 1/20 * sp.integrate.trapezoid(r**2 * w_cont(r)**2, x=r)

    return kappa_u, a_u, connection_point_u, kappa_w, a_w, connection_point_w, Q, fit_points_u, fit_points_w



def Q_integral_trapz_Nmax(Nmax, aHO):
    dr = 0.1
    r = np.arange(0, 100 + dr, dr)
    return sp.integrate.trapezoid(q(r, Nmax, aHO), dx=dr)

def Q_integral_trapz_aHO(Nmax, Nmin, aHO, Nstep):
    Q = []
    Nmax_lst = np.arange(Nmin, Nmax + 1, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(Q_integral_trapz_Nmax, Nmax, aHO) for Nmax in Nmax_lst]
        for future in futures:
            Q.append(future.result())
    
    return np.array([Nmax_lst, Q])

aHO = np.array([1.2, 2.0]) 
r = np.linspace(0, 100, 1000)



def plot_continuation_wavefunc_Q(r, aHO, NNmax, NNmin, dNmax, xmax, fit_point_num1, fit_point_num2, save=False):
    Nmax_lst = np.arange(NNmin, NNmax + 1, dNmax)
    
    plt.figure(figsize=(17, 10.6))
    # plt.subplot(121)

    Q_lst = []
    counter = 0
    plot_wavefunc = 0
    for Nmax in Nmax_lst:
        kappa_u , a_u, connect_u, kappa_w , a_w, connect_w, Q, fit_points_u, fit_points_w = continue_q_wavefunc(Nmax, aHO, fit_point_num1, fit_point_num2)
        
        r_cont_u = np.arange(connect_u, xmax+0.1, 0.1)
        r_cont_w = np.arange(connect_w, xmax+0.1, 0.1)

        u_plot = plt.plot(r, u(r, Nmax, aHO))
        u_color = u_plot[0].get_color()
        plt.plot(r_cont_u, rh0_model_a(r_cont_u, a_u, kappa_u), '--', color=u_color)

        w_plot = plt.plot(r, w(r, Nmax, aHO))
        w_color = w_plot[0].get_color()
        plt.plot(r_cont_w, rh2_model_a(r_cont_w, a_w, kappa_w), "--", color=w_color)
        plt.scatter(fit_points_u[0], fit_points_u[1])
        plt.scatter(fit_points_w[0], fit_points_w[1])
        plt.semilogy()
        plt.grid(linestyle='--')

        Q_lst.append(Q)
        counter += 1
        print(f"Nmax = {Nmax} ({100*counter/len(Nmax_lst)} %)")
        print(f'kappa_u = {kappa_u}, kappa_w = {kappa_w}')
    
    plt.title("$u(r)$ und $w(r)$ sowie deren Fortzetzung für aHO = " + str(aHO) + " fm (logaryhtmische y-Axhse)") 
    plt.ylim(10**(-6), 10)
    plt.xlim(0, xmax)
    plt.xlabel('r (fm)')
    plt.ylabel("$|r\Phi(r)|$")
    plt.legend(["u", "u continuation", "w", "w continuation"])

    if save:
        plt.savefig(f"q_wavefunc_continuation_wavefuncs_aHO{aHO}_NNmax{NNmax}_Nstep{dNmax}_fitted_kappa.pdf")
    
    Q_full_wave = Q_integral_trapz_aHO(NNmax, NNmin, aHO, dNmax)
    # plt.subplot(122)
    plt.figure(figsize=(12, 7.45))
    plt.plot(Q_full_wave[0], Q_full_wave[1], 'x-')
    plt.plot(Nmax_lst, Q_lst, 'x-')
    plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.2709, 0.3, 0.35, 0.4])
    plt.title('$Q$ bei $a_{HO} = $' + str(aHO) + ' fm aus Wellenfunktion und Fortsetzung')
    plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
    plt.xlabel('$N_{max}$')
    plt.ylabel("Q ($e_0$fm$^2$)")
    plt.grid(linestyle='--')

    if save:
        plt.savefig(f"q_wavefunc_continuation_rrms_aHO{aHO}_NNmax{NNmax}_Nstep{dNmax}_fitted_kappa.pdf")

plot_continuation_wavefunc_Q(r, 1.2, 60, 1, 1, 29, 10, 50, False)
# plot_continuation_wavefunc_Q(r, 2.0, 60, 3, 1, 46, 10, 50, False)
# plot_continuation_wavefunc_Q(r, 1.2, 15, 3, 1, 17.5, True)
# plot_continuation_wavefunc_Q(r, 2.0, 15, 3, 1, 27, False)
# plot_continuation_wavefunc_Q(r, 2.4, 60, 3, 1, 60, 80, 40, True)
# plot_continuation_wavefunc_Q(r, 2.6, 60, 3, 1, 65, 80, 40, True)

# print(np.mean([Q_integral_trapz_Nmax(150, aho) for aho in [1.2, 1.4, 1.6, 1.8, 2.0]]))


plt.show()
    
    
