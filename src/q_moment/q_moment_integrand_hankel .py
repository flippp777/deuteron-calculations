import scipy as sp
import numpy as np
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import curve_fit

if sys.platform == "win32":
    sys.path.insert(1, r'C:\Users\Phili\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc

def derivative(x, f, n):
    dx = abs(x[0] - x[1])
    
    for i in range(n):
        f = np.gradient(f) / dx
        
    return f

def u(r, Nmax, aHO):
    return np.abs(r * deuteron_wavefunc(Nmax, aHO, r)[1])

def w(r, Nmax, aHO):
    return np.abs(r * deuteron_wavefunc(Nmax, aHO, r)[2])

def h0_model(r, kappa):
    return 1 / (kappa * r) * np.exp(-kappa * r)

def h0_model_a(r, a, kappa):
    return a / (kappa * r) * np.exp(-kappa * r)

def h2_model(r, kappa):
    return 1/(kappa * r) * np.exp(-kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2)

def h2_model_a(r, a, kappa):
    return a/(kappa * r) * np.exp(-kappa * r) * (1 + 3/(kappa * r) + 3/(kappa * r)**2)

def q(r, Nmax, aHO):
    return 1/np.sqrt(50) * r**2 * u(r, Nmax, aHO) * w(r, Nmax, aHO) - 1/20 * r**2 * w(r, Nmax, aHO)**2

def q_model(r, a_h0, a_h2, kappa):
    return 1/np.sqrt(50) * r**2 * h0_model_a(r, a_h0, kappa) * h2_model_a(r, a_h2, kappa) - 1/20 * r**2 * h2_model_a(r, a_h2, kappa)**2

def q_model_a(r, a, a_h0, a_h2, kappa):
    return a * (1/np.sqrt(50) * r**2 * h0_model_a(r, a_h0, kappa) * h2_model_a(r, a_h2, kappa) - 1/20 * r**2 * h2_model_a(r, a_h2, kappa)**2)

def Q_trapz(r, Nmax, aHO):
    return sp.integrate.trapezoid(q(r, Nmax, aHO), x=r)


def continue_q_integrand(Nmax, aHO):
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
    
    q_lst = np.array([q(r, Nmax - 1, aHO), q(r, Nmax, aHO)])
    
    start_index = np.where(q_lst[1] == max(q_lst[1]))[0][0]
    diff_q = np.abs(q_lst[0, start_index:] - q_lst[1, start_index:])
    
    distance = 0.01
    end_index = np.array([])
    while end_index.size < 3:
        distance += distance + 10**(-6)
        end_index = np.append(end_index,np.where(diff_q < distance)[0] + start_index).flatten()

    end_index = int(max(end_index)) + 1
    fit_points_indices = np.arange(start_index, end_index)
    
    fit_point_num = 30
    if fit_points_indices.size > fit_point_num:
        middle_index = int(fit_points_indices.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices.size / 4)
        fit_points_indices = fit_points_indices[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]
    
    fit_points = [r[fit_points_indices], q_lst[1, fit_points_indices]]
    plt.figure()
    plt.scatter(fit_points[0], fit_points[1])

    # finding the slope with the mean of the derivative
    try:
        a_q, a_h0, a_h2, kappa = curve_fit(q_model_a, fit_points[0], fit_points[1], [1, 1, 1, 0.2307])[0]
        
        # Finding a point that has the same slope as the found one 
        d_dr_q = derivative(r[fit_points_indices], q_lst[1, fit_points_indices], 1)
        q_model_lst = q_model(r[fit_points_indices], a_h0, a_h2, kappa)
        d_dr_q_model = derivative(r[fit_points_indices], q_model_lst)
        connection_point = np.array([])
        delta = 10**(-6)
        while connection_point.size == 0:
            connection_point = r[fit_points_indices][((d_dr_q > d_dr_q_model - delta) & (d_dr_q < d_dr_q_model + delta))]
            delta += 10**(-3)
        connection_point = connection_point[0]

        # Finding the y intersection
        a_q = q(connection_point, Nmax , aHO) / q_model(connection_point, a_h0, a_h2, kappa)
        

        r_wave = r[r <= connection_point]
        r_cont = r[r > connection_point]
        
        q_lst = q(r_wave, Nmax, aHO)
        cont_lst = q_model_a(r, a_q, a_h0, a_h2, kappa)
        
        Q_wave = sp.integrate.trapezoid(q_lst, x=r_wave) 
        Q_cont = sp.integrate.trapezoid(cont_lst, x=r_cont)
        
        Q = Q_wave + Q_cont

        return a_q, a_h0, a_h2, kappa, connection_point, Q, fit_points
    except:
        print('fit error')
        return 0


def Q_integral_trapz_Nmax(Nmax, aHO):
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
    return sp.integrate.trapezoid(q(r, Nmax, aHO), dx=dr)

def Q_integral_trapz_aHO(Nmax, Nmin, aHO, Nstep):
    Q = []
    Nmax_lst = np.arange(Nmin, Nmax + 1, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(Q_integral_trapz_Nmax, Nmax, aHO) for Nmax in Nmax_lst]
        for future in futures:
            Q.append(future.result())
    
    return np.array([Nmax_lst, Q])

aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0]) 
r = np.linspace(0, 100, 1000)

def plot_continuation_integrand_Q(r, aHO, NNmax, NNmin, dNmax, xmax, save=False):
    Nmax_lst = np.arange(NNmin, NNmax + 1, dNmax)
    
    plt.figure(figsize=(17, 10.6))
    # plt.subplot(121)

    Q_lst = []
    err_fit = []
    for Nmax in Nmax_lst:
        try:
            a_q, a_h0, a_h2, kappa, connect, Q, fit_points = continue_q_integrand(Nmax, aHO)

            r_cont = np.arange(connect, xmax+0.1, 0.1)

            q_plt = plt.plot(r, q(r, Nmax, aHO))
            q_plt_color = q_plt[0].get_color()
            plt.plot(r_cont, q_model_a(r_cont, a_q, a_h0, a_h2, kappa), '--', color=q_plt_color)
            # plt.scatter(fit_points[0], fit_points[1])
            plt.semilogy()
            plt.grid(linestyle='--')
            Q_lst.append(Q)
        except:
            err_fit.append(np.where(Nmax == Nmax_lst)[0][0])
    
    plt.title("$1/\sqrt{50} r^2 u(r) w(r) - 1/20 r^2 w(r)^2$ und Fortzetzung für aHO = " + str(aHO) + " fm (logaryhtmische y-Axhse)") 
    plt.ylim(10**(-6), 1)
    plt.xlim(0, 20)
    plt.xlabel('r (fm)')
    plt.legend(["$1/\sqrt{50} r^2 u(r) w(r) - 1/20 r^2 w(r)^2$", "Fortsetzung"])
    
    if save:
        plt.savefig(f"q_integrand_continuation_rrms_aHO{aHO}_NNmax{NNmax}_Nstep{dNmax}_fitted_kappa.pdf")

    Q_full_wave = Q_integral_trapz_aHO(NNmax, NNmin, aHO, dNmax)
    # plt.subplot(122)
    plt.figure(figsize=(12, 7.45))
    plt.plot(Q_full_wave[0], Q_full_wave[1], 'x-')
    plt.plot(np.delete(Nmax_lst, err_fit), Q_lst, 'x-')
    plt.ylim(0.15, 0.4)
    plt.yticks([0.15, 0.2, 0.25, 0.2709, 0.3, 0.35, 0.4, 0.45])
    plt.title('El. Quadrupolmoment bei $a_{HO} = $' + str(aHO) + ' fm aus Wellenfunktion und Fortsetzung')
    plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
    plt.xlabel('$N_{max}$')
    plt.ylabel("$Q_{d}$ ($e_{0}$fm$^2$)")
    plt.grid(linestyle='--')

    if save:
        plt.savefig(f"q_integrand_continuation_rrms_aHO{aHO}_NNmax{NNmax}_Nstep{dNmax}_fitted_kappa.pdf")


# plot_continuation_integrand_Q(r, 60, 5, 2)
plot_continuation_integrand_Q(r, 1.2, 60, 3, 10, 29)
plot_continuation_integrand_Q(r, 2.0, 60, 3, 10, 29)

plt.show()
    
    
