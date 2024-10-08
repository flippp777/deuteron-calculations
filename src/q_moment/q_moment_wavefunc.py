import scipy as sp
import numpy as np
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')
from make.position_space_wavefunc import deuteron_wavefunc

def log_model(r, a, b):
    return a*r+b

def derivative(x, f, n):
    dx = abs(x[0] - x[1])
    
    for i in range(n):
        f = np.gradient(f) / dx
        
    return f

def u(r, Nmax, aHO):
    return np.abs(r * deuteron_wavefunc(Nmax, aHO, r)[1])

def w(r, Nmax, aHO):
    return np.abs(r * deuteron_wavefunc(Nmax, aHO, r)[2])

def Q_quad(Nmax, aHO):
    return sp.integrate.quad(lambda r: 1/np.sqrt(50) * r**2 * u(r, Nmax, aHO) * w(r, Nmax, aHO) - 1/20 * r**2 * w(r, Nmax, aHO)**2, 0, 100)[0]

def q(r, Nmax, aHO):
    return 1/np.sqrt(50) * r**2 * u(r, Nmax, aHO) * w(r, Nmax, aHO) - 1/20 * r**2 * w(r, Nmax, aHO)**2

def Q_trapz(r, Nmax, aHO):
    return sp.integrate.trapezoid(q(r, Nmax, aHO), x=r)


def continue_q_integrand(Nmax, aHO):
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
    
    q_lst_0 = q(r, Nmax - 1, aHO)
    log_q_lst_0 = np.log(q_lst_0)
    q_lst_1 = q(r, Nmax, aHO)
    log_q_lst_1 = np.log(q_lst_1)
    
    nan_index = np.where(np.isnan(log_q_lst_0)[0])
    nan_index = np.append(nan_index, np.where(np.isnan(log_q_lst_1))[0]).flatten()
    
    r = np.delete(r, nan_index)
    q_lst_0 = np.delete(q_lst_0, nan_index)
    q_lst_1 = np.delete(q_lst_1, nan_index)
    log_q_lst_1 = np.delete(log_q_lst_1, nan_index)
    log_q_lst_0 = np.delete(log_q_lst_0, nan_index)
    
    q_lst = np.array([q_lst_0, q_lst_1])
    log_q = np.log(q_lst)
    
    start_index = np.where(log_q[1] == max(log_q[1]))[0][0]
    diff_log_q = np.abs(log_q[0, start_index:] - log_q[1, start_index:])
    
    distance = 0.01
    end_index = np.array([])
    while end_index.size < 3:
        distance += distance + 10**(-6)
        end_index = np.append(end_index,np.where(diff_log_q < distance)[0] + start_index).flatten()

    end_index = int(max(end_index)) + 1
    fit_points_indices = np.arange(start_index, end_index)
    
    fit_point_num = 20
    if fit_points_indices.size > fit_point_num:
        middle_index = int(fit_points_indices.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices.size / 4)
        fit_points_indices = fit_points_indices[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]
    
    # finding the slope with the mean of the derivative
    slope = np.mean(derivative(r[fit_points_indices], log_q[1, fit_points_indices], 1))

    fit_points = [r[fit_points_indices], log_q[1, start_index: end_index]]
    
    # Finding a point that has the same slope as the found one 
    d_dr = derivative(r, log_q[1, fit_points_indices], 1)
    connection_point = np.array([])
    delta = 10**(-6)
    while connection_point.size == 0:
        connection_point = r[fit_points_indices][((d_dr > slope - delta) & (d_dr < slope + delta))]
        delta += 10**(-3)
    connection_point = connection_point[0]

    # Finding the y intersection
    y_intersect = np.log(q(connection_point, Nmax , aHO)) - slope * connection_point
    

    r_wave = r[r <= connection_point]
    r_cont = r[r > connection_point]
    
    q_lst = q(r_wave, Nmax, aHO)
    cont_lst = np.exp(log_model(r_cont, slope, y_intersect))
    
    Q_wave = sp.integrate.trapezoid(q_lst, x=r_wave) 
    Q_cont = sp.integrate.trapezoid(cont_lst, x=r_cont)
    
    Q = Q_wave + Q_cont

    return slope, y_intersect, connection_point, Q, fit_points


def continue_q_wavefunc(Nmax, aHO):
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
    
    u_lst = np.array([u(r, Nmax-1, aHO), u(r, Nmax, aHO)])
    w_lst = np.array([w(r, Nmax-1, aHO), w(r, Nmax, aHO)])
    
    log_u = np.log(u_lst)
    log_w = np.log(w_lst)
    
    start_index_u = np.where(log_u[1] == max(log_u[1]))[0][0]
    diff_log_u = np.abs(log_u[0, start_index_u:] - log_u[1, start_index_u:])
    start_index_w = np.where(log_w[1] == max(log_w[1]))[0][0]
    diff_log_w = np.abs(log_w[0, start_index_w:] - log_w[1, start_index_w:])
    
    distance = 0.01
    end_index_u = np.array([])
    while end_index_u.size < 3:
        distance += distance + 10**(-6)
        end_index_u = np.append(end_index_u,np.where(diff_log_u < distance)[0] + start_index_u).flatten()
    distance = 0.01
    end_index_w = np.array([])
    while end_index_w.size < 3:
        distance += distance + 10**(-6)
        end_index_w = np.append(end_index_w,np.where(diff_log_w < distance)[0] + start_index_w).flatten()

    end_index_u = int(max(end_index_u)) + 1
    fit_points_indices_u = np.arange(start_index_u, end_index_u)
    end_index_w = int(max(end_index_w)) + 1
    fit_points_indices_w = np.arange(start_index_w, end_index_w)
    
    fit_point_num = 10
    if fit_points_indices_u.size > fit_point_num:
        middle_index = int(fit_points_indices_u.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices_u.size / 4)
        fit_points_indices_u = fit_points_indices_u[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]
    if fit_points_indices_w.size > fit_point_num:
        middle_index = int(fit_points_indices_u.size / 2)
        # two_third_index = int(fit_points_indices.size * 2 / 3)
        three_quater_index = middle_index + int(fit_points_indices_w.size / 4)
        fit_points_indices_w = fit_points_indices_w[three_quater_index - int(fit_point_num / 2):three_quater_index + int(fit_point_num / 2)]
    
    # finding the slope with the mean of the derivative
    fit_points_u = np.array([r[fit_points_indices_u], log_u[1, fit_points_indices_u]])
    fit_points_w = np.array([r[fit_points_indices_w], log_w[1, fit_points_indices_w]])
    slope_u = np.mean(derivative(fit_points_u[0], fit_points_u[1], 1))
    slope_w = np.mean(derivative(fit_points_w[0], fit_points_w, 1))

    # Finding a point that has the same slope as the found one 
    d_dr_u = derivative(r, log_u[1, fit_points_indices_u], 1)
    connection_point_u = np.array([])
    delta = 10**(-6)
    while connection_point_u.size == 0:
        connection_point_u = r[fit_points_indices_u][((d_dr_u > slope_u - delta) & (d_dr_u < slope_u + delta))]
        delta += 10**(-3)
    connection_point_u = connection_point_u[0]
    d_dr_w = derivative(r, log_w[1, fit_points_indices_w], 1)
    connection_point_w = np.array([])
    delta = 10**(-6)
    while connection_point_w.size == 0:
        connection_point_w = r[fit_points_indices_w][((d_dr_w > slope_w - delta) & (d_dr_w < slope_w + delta))]
        delta += 10**(-3)
    connection_point_w = connection_point_w[0]

    # Finding the y intersection
    y_intersect_u = np.log(q(connection_point_u, Nmax , aHO)) - slope_u * connection_point_u
    y_intersect_w = np.log(q(connection_point_w, Nmax , aHO)) - slope_w * connection_point_w
    
    def u_cont(r):
        u_cont = []
        for i in range(r.size):
            if r[i] < connection_point_u:
                u_cont.append(u_lst[1, i])
            else:
                u_cont.append(np.exp(log_model(r[i], slope_u, y_intersect_u)))
        return np.array(u_cont)
    def w_cont(r):
        w_cont = []
        for i in range(r.size):
            if r[i] < connection_point_w:
                w_cont.append(w_lst[1, i])
            else:
                w_cont.append(np.exp(log_model(r[i], slope_w, y_intersect_w)))
        return np.array(w_cont)

    Q = 1/np.sqrt(50) * sp.integrate.trapezoid(r**2 * u_cont(r) * w_cont(r), x=r) - 1/20 * sp.integrate.trapezoid(r**2 * w_cont(r)**2, x=r)

    return slope_u, y_intersect_u, connection_point_u, slope_w, y_intersect_w, connection_point_w, Q, fit_points_u, fit_points_w


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

def plot_continuation_wavefunc_Q(r, NNmax, NNmin, dNmax):
    Nmax_lst = np.arange(NNmin, NNmax + 1, dNmax)
    
    for j in range(aHO_lst.size):
        print(str(20 * j) + " %")
        
        plt.figure()
        plt.subplot(121)

        Q_lst = []
        for Nmax in Nmax_lst:
            slope , y_sect, connect, Q, fit_points = continue_q_integrand(Nmax, aHO_lst[j])
            r_cont = np.array([connect, 20])
            plt.plot(r, q(r, Nmax, aHO_lst[j]), 'b')
            plt.plot(r_cont, np.exp(log_model(r_cont, slope, y_sect)), "r")
            # plt.scatter(fit_points[0], np.exp(fit_points[1]))
            plt.semilogy()
            plt.grid(linestyle='--')
            Q_lst.append(Q)
        
        plt.title("Wellenfunktionen und exponentielle Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 20)
        plt.legend(["Wellenfunktion", "Fortsetzung"])
        
        Q_full_wave = Q_integral_trapz_aHO(NNmax, NNmin, aHO_lst[j], dNmax)
        plt.subplot(122)
        plt.plot(Q_full_wave[0], Q_full_wave[1], 'x-')
        plt.plot(Nmax_lst, Q_lst, 'x-')
        plt.ylim(0, 0.4)
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und exponentieller Fortsetzung')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.grid(linestyle='--')


def plot_continuation_wavefunc_Q(r, NNmax, NNmin, dNmax):
    Nmax_lst = np.arange(NNmin, NNmax + 1, dNmax)
    
    for j in range(aHO_lst.size):
        print(str(20 * j) + " %")
        
        plt.figure()
        plt.subplot(121)

        Q_lst = []
        for Nmax in Nmax_lst:
            slope_u , y_sect_u, connect_u, slope_w , y_sect_w, connect_w, Q, fit_points_u, fit_points_w = continue_q_wavefunc(Nmax, aHO_lst[j])
            r_cont_u = np.array([connect_u, 20])
            plt.plot(r, u(r, Nmax, aHO_lst[j]), 'b')
            plt.plot(r_cont_u, np.exp(log_model(r_cont_u, slope_u, y_sect_u)), "r")
            r_cont_w = np.array([connect_w, 20])
            plt.plot(r, w(r, Nmax, aHO_lst[j]), 'c')
            plt.plot(r_cont_w, np.exp(log_model(r_cont_w, slope_w, y_sect_w)), "m")
            # plt.scatter(fit_points[0], np.exp(fit_points[1]))
            plt.semilogy()
            plt.grid(linestyle='--')
            Q_lst.append(Q)
        
        plt.title("Wellenfunktionen und exponentielle Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 20)
        plt.legend(["u", "u continuation", "w", "w continuation"])
        
        Q_full_wave = Q_integral_trapz_aHO(NNmax, NNmin, aHO_lst[j], dNmax)
        plt.subplot(122)
        plt.plot(Q_full_wave[0], Q_full_wave[1], 'x-')
        plt.plot(Nmax_lst, Q_lst, 'x-')
        plt.ylim(0, 0.4)
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und exponentieller Fortsetzung')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.grid(linestyle='--')

# plot_continuation_integrand_Q(r, 60, 5, 2)
plot_continuation_wavefunc_Q(r, 10, 5, 2)

plt.show()
    
    
