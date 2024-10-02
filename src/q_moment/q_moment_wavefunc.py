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
    return r * deuteron_wavefunc(Nmax, aHO, r)[1]

def w(r, Nmax, aHO):
    return r * deuteron_wavefunc(Nmax, aHO, r)[2]

def Q_quad(Nmax, aHO):
    return sp.integrate.quad(lambda r: 1/np.sqrt(50) * r**2 * u(r, Nmax, aHO) * w(r, Nmax, aHO) - 1/20 * r**2 * w(r, Nmax, aHO)**2, 0, 100)[0]

def q(r, Nmax, aHO):
    return 1/np.sqrt(50) * r**2 * u(r, Nmax, aHO) * w(r, Nmax, aHO) - 1/20 * r**2 * w(r, Nmax, aHO)**2

def Q_trapz(r, Nmax, aHO):
    return sp.integrate.trapezoid(q(r, Nmax, aHO), x=r)

def continue_q(Nmax, aHO):
    dr = 0.1
    r = np.arange(0, 50 + dr, dr)
    
    q_lst = np.array([q(r, Nmax - 1, aHO), q(r, Nmax, aHO)])
    log_q = np.log(q_lst)
    
    start_index = np.where(log_q[1] == max(log_q[1]))[0][0]
    diff_log_q = np.abs(log_q[0, start_index:] - log_q[1, start_index:])
    
    distance = 0.01
    end_index = np.array([])
    while end_index.size < 3:
        distance += distance + 10**(-6)
        end_index = np.append(end_index,np.where(diff_log_q < distance)[0] + start_index).flatten()

    end_index = int(max(end_index))
    
    # finding the slope with the mean of the derivative
    slope = np.mean(derivative(r, log_q[1, start_index: end_index], 1))

    fit_points = [r[start_index: end_index], log_q[1, start_index: end_index]]
    
    # Finding a point that has the same slope as the found one 
    d_dr = derivative(r, log_q[1, start_index: end_index], 1)
    connection_point = np.array([])
    delta = 10**(-6)
    while connection_point.size == 0:
        connection_point = r[start_index: end_index][((d_dr > slope - delta) & (d_dr < slope + delta))]
        delta += 10**(-6)
    connection_point = connection_point[0]
    
    # Finding the y intersection
    y_intersect = np.log(q(connection_point, Nmax , aHO)) - slope * connection_point
    

    r_wave = r[r <= connection_point]
    r_cont = r[r > connection_point]
    
    q_lst = q(r_wave, Nmax, aHO)
    cont_lst = np.exp(log_model(r_cont, slope, y_intersect))
    
    Q_wave = sp.integrate.trapezoid(q_lst, x=r_wave) / 4
    Q_cont = sp.integrate.trapezoid(cont_lst, x=r_cont) / 4
    
    Q = Q_wave + Q_cont

    return slope, y_intersect, connection_point, Q, fit_points


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

def plot_continuation_Q(r, NNmax, NNmin, dNmax):
    Nmax_lst = np.arange(NNmin, NNmax + 1, dNmax)
    
    for j in range(aHO_lst.size):
        plt.figure()
        plt.subplot(121)

        rrms_lst = []
        for Nmax in Nmax_lst:
            slope , y_sect, connect, rrms, fit_points = continue_q(Nmax, aHO_lst[j])
            r_cont = np.array([connect, 20])
            plt.plot(r, q(r, Nmax, aHO_lst[j]), 'b')
            plt.plot(r_cont, np.exp(log_model(r_cont, slope, y_sect)), "r")
            # plt.scatter(fit_points[0], np.exp(fit_points[1]))
            plt.semilogy()
            plt.grid(linestyle='--')
            rrms_lst.append(rrms)
        
        plt.title("Wellenfunktionen und exponentielle Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 20)
        plt.legend(["Wellenfunktion", "Fortsetzung"])
        
        rrms_full_wave = Q_integral_trapz_aHO(NNmax, NNmin-1, aHO_lst[j], dNmax)
        plt.subplot(122)
        plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
        plt.plot(Nmax_lst, rrms_lst, 'x-')
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und exponentieller Fortsetzung')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
        plt.grid(linestyle='--')


plot_continuation_Q(r, 40, 1, 40)
plt.show()
    
    
