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
from plotstyle.colorFader import colorFader
from plotstyle.set_size import set_size

plt.style.use('src\stylesheet.mplstyle')

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
        delta = distance + 10**(-3)
        while np.unique(fit_points_indices_u).size < number_at_least_collected_points:
            fit_points_indices_u = np.append(fit_points_indices_u, np.where(diff_log_u < delta)[0] + start_index_u).flatten()
            delta += 10**(-3)

    if np.unique(fit_points_indices_w).size < number_at_least_collected_points:
        delta = distance + 10**(-3)
        while np.unique(fit_points_indices_w).size < number_at_least_collected_points:
            fit_points_indices_w = np.append(fit_points_indices_w,np.where(diff_log_w < delta)[0] + start_index_w).flatten()
            delta += 10**(-3)

    fit_points_indices_u = np.unique(fit_points_indices_u)
    fit_points_indices_w = np.unique(fit_points_indices_w)
    
    cutoff_index = 5 
    if fit_points_indices_u.size > number_used_points:
        middle_index = int(fit_points_indices_u.size / 2)
        three_quater_index = middle_index + int(fit_points_indices_u.size / 4)
        fit_points_indices_u = fit_points_indices_u[three_quater_index - int(number_used_points / 2):three_quater_index + int(number_used_points / 2)]
        # fit_points_indices_u = fit_points_indices_u[-number_used_points-cutoff_index:-cutoff_index]
    if fit_points_indices_w.size > number_used_points:
        middle_index = int(fit_points_indices_w.size / 2)
        three_quater_index = middle_index + int(fit_points_indices_w.size / 4)
        fit_points_indices_w = fit_points_indices_w[three_quater_index - int(number_used_points / 2):three_quater_index + int(number_used_points / 2)]
        # fit_points_indices_w = fit_points_indices_w[-number_used_points-cutoff_index:-cutoff_index]
    
    # finding the slope with the mean of the derivative
    fit_points_u = np.array([r[fit_points_indices_u], u_lst[1, fit_points_indices_u]])
    fit_points_w = np.array([r[fit_points_indices_w], w_lst[1, fit_points_indices_w]])

    fit_point_mean_u = np.mean(fit_points_u[1])
    fit_point_std_u = np.std(fit_points_u[1])
    fit_points_u = fit_points_u.transpose()[(
        (fit_points_u[1] <= fit_point_mean_u + 1.5 * fit_point_std_u) & 
        (fit_points_u[1] >= fit_point_mean_u - 1.5 * fit_point_std_u))].transpose()

    fit_point_mean_w = np.mean(fit_points_w[1])
    fit_point_std_w = np.std(fit_points_w[1])
    fit_points_w = fit_points_w.transpose()[(
        (fit_points_w[1] <= fit_point_mean_w + 1.5 * fit_point_std_w) & 
        (fit_points_w[1] >= fit_point_mean_w - 1.5 * fit_point_std_w))].transpose()

    # Fit
    params_u = create_params(a=1, kappa=dict(value=0.2305, min=0.1, max=0.4))
    
    h0model = Model(rh0_model_a)
    h0result = h0model.fit(fit_points_u[1], r=fit_points_u[0], params=params_u)
    kappa_u = h0result.best_values["kappa"]
    
    params_w = create_params(a=0.01, kappa=dict(value=0.2305, min=0.1, max=0.4))
    
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
    rh0 = rh0_model(fit_points_u[0], kappa_u)
    d_dr_rh0 = np.gradient(np.log(rh0), fit_points_u[0])[1:-1]
    connection_point_u = np.array([])
    delta = 10**(-3)
    while connection_point_u.size == 0:
        connection_point_u = fit_points_u[0][1:-1][((d_dr_u > d_dr_rh0 - delta) & (d_dr_u < d_dr_rh0 + delta))]
        delta += 10**(-3)
    connection_point_u = connection_point_u[0]
    
    d_dr_w = np.gradient(np.log(fit_points_w[1]), fit_points_w[0])[1:-1]
    rh2 = rh2_model(fit_points_w[0], kappa_w)
    d_dr_rh2 = np.gradient(np.log(rh2), fit_points_w[0])[1:-1]
    connection_point_w = np.array([])
    delta = 10**(-3)
    while connection_point_w.size == 0:
        connection_point_w = fit_points_w[0][1:-1][((d_dr_w > d_dr_rh2 - delta) & (d_dr_w < d_dr_rh2 + delta))]
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
    r = np.arange(0, 250 + dr, dr)
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
    kappa_u_lst = []
    kappa_w_lst = []
    counter = 0
    plot_wavefunc = 0
    for Nmax in Nmax_lst:
        kappa_u , a_u, connect_u, kappa_w , a_w, connect_w, Q, fit_points_u, fit_points_w = continue_q_wavefunc(Nmax, aHO, fit_point_num1, fit_point_num2)
        
        r_cont_u = np.arange(connect_u, xmax+0.1, 0.1)
        r_cont_w = np.arange(connect_w, xmax+0.1, 0.1)

        u_plot = plt.plot(r, u(r, Nmax, aHO))
        u_color = u_plot[0].get_color()
        plt.plot(r_cont_u, rh0_model_a(r_cont_u, a_u, kappa_u), '--', color=u_color)
        plt.scatter(fit_points_u[0], fit_points_u[1], color=u_color)

        w_plot = plt.plot(r, w(r, Nmax, aHO))
        w_color = w_plot[0].get_color()
        plt.plot(r_cont_w, rh2_model_a(r_cont_w, a_w, kappa_w), "--", color=w_color)
        plt.scatter(fit_points_w[0], fit_points_w[1], color=w_color)
        plt.semilogy()
        plt.grid(linestyle='--')

        Q_lst.append(Q)
        kappa_u_lst.append(round(kappa_u,4))
        kappa_w_lst.append(round(kappa_w,4))
        counter += 1
        print(f"Nmax = {Nmax} ({100*counter/len(Nmax_lst)} %)")
        print(f'kappa_u = {kappa_u}, kappa_w = {kappa_w}')

    # np.savetxt(f'uw_fitted_kappas_aHO{aHO}.txt', np.array([Nmax_lst, kappa_u_lst, kappa_w_lst]).transpose())
    
    plt.title("$u(r)$ und $w(r)$ sowie deren Fortzetzung für aHO = " + str(aHO) + " fm (logaryhtmische y-Axhse)") 
    plt.ylim(10**(-6), 10)
    plt.xlim(0, xmax)
    plt.xlabel('r (fm)')
    plt.ylabel("$|r\Phi(r)|$")
    plt.legend(["u(r)", "u(r) continuation", "w(r)", "w(r) continuation"])
    # plt.legend(np.array([[f"Nmax={Nmax_lst[i]}, u(r)", f"Nmax={Nmax_lst[i]}, u(r)-Fortsetzung, $\kappa$={kappa_u_lst[i]}", f"Nmax={Nmax_lst[i]}, u(r)-Fitpunkte", 
    #                       f"Nmax={Nmax_lst[i]}, w(r)", f"Nmax={Nmax_lst[i]}, w(r)-Fortsetzung, $\kappa$={kappa_w_lst[i]}", f"Nmax={Nmax_lst[i]}, w(r)-Fitpunkte"] for i in range(0,Nmax_lst.size)]).flatten())

    if save:
        plt.savefig(f"q_wavefunc_continuation_wavefuncs_aHO{aHO}_NNmax{NNmax}_Nstep{dNmax}_fitted_kappa.pdf")
    
    Q_full_wave = Q_integral_trapz_aHO(NNmax, NNmin, aHO, dNmax)
    # plt.subplot(122)
    plt.figure(figsize=(12, 7.45))
    plt.plot(Q_full_wave[0], Q_full_wave[1], 'x-')
    plt.plot(Nmax_lst, Q_lst, 'x-')
    plt.yticks(np.append([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.2709], np.arange(0.3, max(Q_lst)+0.05, 0.05)))
    # plt.yticks([0.2, 0.225, 0.25, 0.2709, 0.275, 0.3, 0.325, 0.35])
    plt.title('$Q$ bei $a_{HO} = $' + str(aHO) + ' fm aus Wellenfunktion und Fortsetzung')
    plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
    plt.xlabel('$N_{max}$')
    plt.ylabel("Q ($e_0$fm$^2$)")
    plt.grid(linestyle='--')

    if save:
        plt.savefig(f"q_wavefunc_continuation_q_aHO{aHO}_NNmax{NNmax}_Nstep{dNmax}_fitted_kappa.pdf")

def plot_wavefunc_continuation_q_final(save=False):
    r = np.arange(0, 250, 0.1)
    Nmax_lst = np.arange(1, 30 + 1, 1)
    
    purple = '#FF00FF'
    blue = '#0000FF'
    dark_blue = '#000019'
    red = '#FF0000'
    dark_red = '#190000'
    black = '#000000'
    n= len(Nmax_lst)
 

    aHO_lst = [1.2, 2,0]
    q_lst = [[0],[0]]
    for j in range(2):
        for i in range(n):
            kappa_phi0, c_phi0, connect_phi0, kappa_phi2, c_phi2, connect_phi2, q, fit_points_phi0, fit_points_phi2 = continue_q_wavefunc(Nmax_lst[i], aHO_lst[j], 10, 50)
            q_lst[j].append(q)
    
    q_12 = np.delete(q_lst[0],0) 
    q_20 = np.delete(q_lst[1],0) 
    
    q_full_wave_12 = Q_integral_trapz_aHO(30, 0, 1.2, 1)
    q_full_wave_20 = Q_integral_trapz_aHO(30, 0, 2.0, 1)
    
    fig = plt.figure(figsize=set_size(502.14963))
    gs = fig.add_gridspec(1, 2, wspace=0, hspace=0)
    axs = gs.subplots(sharey=True)

    axs[0].plot(q_full_wave_12[0], q_full_wave_12[1], 'x-', color='black', markersize=4)
    axs[0].plot(Nmax_lst, q_12, 'x-', color=purple, markersize=4)
    axs[0].legend(["\(Q^\\text{wav}\)", '\(Q^\\text{fit}\)'])
    axs[0].set_xlabel('$\\Tilde{N}_\\text{max}$')
    axs[0].set_ylabel("$Q$ [$e$fm$^2$]")
    axs[0].text(20, 0.175, '$a_\\text{HO}=1.2\ $fm')
    axs[0].set_ylim(0.15,0.45)
    axs[0].set_yticks([0.15, 0.2, 0.25, 0.2709, 0.3, 0.35, 0.4, 0.45])
    axs[0].grid(linestyle='--')
    
    axs[1].plot(q_full_wave_20[0], q_full_wave_20[1], 'x-', color='black', markersize=4)
    axs[1].plot(Nmax_lst, q_20, 'x-', color=purple, markersize=4)
    axs[1].legend(["\(Q^\\text{wav}\)", '\(Q^\\text{fit}\)'])
    axs[1].set_xlabel('$\\Tilde{N}_\\text{max}$')
    axs[1].text(20, 0.175, '$a_\\text{HO}=2.0\ $fm')
    axs[1].grid(linestyle='--')

    if save:
        plt.savefig("q_continuation_1.2-2.0_final_fitted_kappa.pdf", bbox_inches='tight')

def q_values(aHO, save=False):
    Nmax = [2,5,8,11,16,20]
    q_contiunuation = []
    q_wavefunc = [] 
    q_conv = 0.2709
    for i in Nmax:
            kappa_phi0, c_phi0, connect_phi0, kappa_phi2, c_phi2, connect_phi2, q, fit_points_phi0, fit_points_phi2 = continue_q_wavefunc(i, aHO, 10, 50)
            q_contiunuation.append(q)
            q_wavefunc.append(Q_integral_trapz_Nmax(i, aHO))
    
    dev_continuation = np.abs((np.array(q_contiunuation)-q_conv)/q_conv)
    dev_wavefunc = np.abs((np.array(q_wavefunc)-q_conv)/q_conv)
    
    full_arr = np.array([Nmax, q_contiunuation, dev_continuation, q_wavefunc, dev_wavefunc])
    print(full_arr)
    
    if save:
        np.savetxt(f'q_fitted_values_aHO{aHO}.txt', full_arr)
        
# q_values(2, True)

plot_wavefunc_continuation_q_final(True)

# plot_continuation_wavefunc_Q(r, 1.2, 60, 1, 1, 29, 10, 50, False)
# plot_continuation_wavefunc_Q(r, 2.0, 60, 3, 1, 46, 10, 50, False)
# plot_continuation_wavefunc_Q(r, 1.2, 15, 3, 1, 17.5, True)
# plot_continuation_wavefunc_Q(r, 2.0, 15, 3, 1, 27, False)
# plot_continuation_wavefunc_Q(r, 2.4, 60, 3, 1, 60, 80, 40, True)
# plot_continuation_wavefunc_Q(r, 2.6, 60, 3, 1, 65, 80, 40, True)

# print(np.mean([Q_integral_trapz_Nmax(150, aho) for aho in [1.2, 1.4, 1.6, 1.8, 2.0]]))


plt.show()
    
    
