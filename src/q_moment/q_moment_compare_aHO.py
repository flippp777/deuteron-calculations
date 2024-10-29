import numpy as np
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import trapezoid

if sys.platform == "win32":
    sys.path.insert(1, r'C:\Users\Phili\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc


def u(r, Nmax, aHO):
    return np.abs(r * deuteron_wavefunc(Nmax, aHO, r)[1])

def w(r, Nmax, aHO):
    return np.abs(r * deuteron_wavefunc(Nmax, aHO, r)[2])

def q(r, Nmax, aHO):
    return 1/np.sqrt(50) * r**2 * u(r, Nmax, aHO) * w(r, Nmax, aHO) - 1/20 * r**2 * w(r, Nmax, aHO)**2


def Q_trapz(r, Nmax, aHO):
    return trapezoid(q(r, Nmax, aHO), x=r)

def Q_integral_trapz_Nmax(Nmax, aHO):
    dr = 0.1
    r = np.arange(0, 100 + dr, dr)
    return trapezoid(q(r, Nmax, aHO), dx=dr)

def Q_integral_trapz_aHO(Nmax, Nmin, aHO, Nstep):
    Q = []
    Nmax_lst = np.arange(Nmin, Nmax + Nstep, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(Q_integral_trapz_Nmax, Nmax, aHO) for Nmax in Nmax_lst]
        for future in futures:
            Q.append(future.result())
    
    return np.array([Nmax_lst, Q])

aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0])

def plot_Q(Nmax, Nmin, Nstep):
    q_aHO_lst = []
    
    for aHO in aHO_lst:
        q_aHO_lst.append(Q_integral_trapz_aHO(Nmax, Nmin, aHO, Nstep))
        
    q_aHO_lst = np.array(q_aHO_lst)

    print(q_aHO_lst[:,1,-1])
    
    Nmax_lst = np.arange(Nmin, Nmax + Nstep, Nstep)
    q_mean_lst = []
    for i in range(Nmax_lst.size):
        q_mean_lst.append(np.mean([q_aHO_lst[:, 1, i]]))
    
    q_mean_lst = np.array([Nmax_lst, q_mean_lst])
    print(q_mean_lst[1,-1])
    
    plt.figure()

    plt.scatter(q_aHO_lst[0,0], q_aHO_lst[0,1])
    plt.scatter(q_aHO_lst[1,0], q_aHO_lst[1,1])
    plt.scatter(q_aHO_lst[2,0], q_aHO_lst[2,1])
    plt.scatter(q_aHO_lst[3,0], q_aHO_lst[3,1])
    plt.scatter(q_aHO_lst[4,0], q_aHO_lst[4,1])
    plt.plot(q_mean_lst[0], q_mean_lst[1]) 
    plt.grid()

    plt.ylim(0.27, 0.28)    

plot_Q(150, 3, 10)

plt.show()
