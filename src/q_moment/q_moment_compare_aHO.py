import numpy as np
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import trapezoid

if sys.platform == "win32":
    sys.path.insert(1, r'C:\Users\phili.DESKTOP\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc
from plotstyle.colorFader import colorFader
from plotstyle.set_size import set_size


def absu(r, Nmax, aHO):
    return np.abs(r * deuteron_wavefunc(Nmax, aHO, r)[1])

def absw(r, Nmax, aHO):
    return np.abs(r * deuteron_wavefunc(Nmax, aHO, r)[2])

def u(r, Nmax, aHO):
    if deuteron_wavefunc(Nmax, aHO, 2)[1] > 0:
        return r * deuteron_wavefunc(Nmax, aHO, r)[1]
    else:
        return -r * deuteron_wavefunc(Nmax, aHO, r)[1]

def w(r, Nmax, aHO):
    if deuteron_wavefunc(Nmax, aHO, 2)[2] > 0:
        return r * deuteron_wavefunc(Nmax, aHO, r)[2]
    else:
        return -r * deuteron_wavefunc(Nmax, aHO, r)[2]

def absq(r, Nmax, aHO):
    return 1/np.sqrt(50) * r**2 * absu(r, Nmax, aHO) * absw(r, Nmax, aHO) - 1/20 * r**2 * absw(r, Nmax, aHO)**2

def q(r, Nmax, aHO):
    return 1/np.sqrt(50) * r**2 * u(r, Nmax, aHO) * w(r, Nmax, aHO) - 1/20 * r**2 * w(r, Nmax, aHO)**2


def Q_trapz(r, Nmax, aHO):
    return trapezoid(absq(r, Nmax, aHO), x=r)

def Q_integral_trapz_Nmax(Nmax, aHO):
    dr = 0.1
    r = np.arange(0, 100 + dr, dr)
    return trapezoid(absq(r, Nmax, aHO), dx=dr)

def Q_integral_trapz_aHO(Nmax, Nmin, aHO, Nstep):
    Q = []
    Nmax_lst = np.arange(Nmin, Nmax + Nstep, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(Q_integral_trapz_Nmax, Nmax, aHO) for Nmax in Nmax_lst]
        for future in futures:
            Q.append(future.result())
    
    return np.array([Nmax_lst, Q])

def Q_integral_trapz_Nmax_normal(Nmax, aHO):
    dr = 0.1
    r = np.arange(0, 100 + dr, dr)
    return trapezoid(q(r, Nmax, aHO), dx=dr)

def Q_integral_trapz_aHO_normal(Nmax, Nmin, aHO, Nstep):
    Q = []
    Nmax_lst = np.arange(Nmin, Nmax + Nstep, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(Q_integral_trapz_Nmax_normal, Nmax, aHO) for Nmax in Nmax_lst]
        for future in futures:
            Q.append(future.result())
    
    return np.array(Q)

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

def final_plot(save=False):
    Nmax = np.arange(0, 31, 1)
    Q_abs=[]
    for i in aHO_lst:
        Q_abs.append(Q_integral_trapz_aHO(30, 0, i, 1)[1])
    Q=[]
    for i in aHO_lst:
        Q.append(Q_integral_trapz_aHO_normal(30, 0, i, 1))
    
    fig = plt.figure(figsize=set_size(502.14963))
    gs = fig.add_gridspec(1, 2, wspace=0, hspace=0)
    axs = gs.subplots(sharey=True)

    for i in range(aHO_lst.shape[0]):
        axs[0].plot(Nmax, Q[i], 'x-', color=colorFader('#FF00FF', '#000000', i/len(aHO_lst)), markersize=4)
    legend = np.array([['$a_{HO}$=' + str(aHO)] for aHO in aHO_lst]).flatten()
    axs[0].legend(legend, loc='upper right', ncol=1)
    axs[0].set_ylabel("$Q$ ($e_0$fm$^2$)")
    axs[0].set_yticks([0.15, 0.20, 0.25, 0.271, 0.30, 0.35, 0.40, 0.45])
    axs[0].set_xlabel('$n_{max}$')
    # axs[0].set_xlim(0, 30)
    axs[0].grid(linestyle='--')

    for i in range(aHO_lst.shape[0]):
        plt.plot(Nmax, Q_abs[i], 'x-', color=colorFader('#FF00FF', '#000000', i/len(aHO_lst)), markersize=4)
    axs[1].legend(legend, loc='upper right', ncol=1)
    axs[1].set_xlabel('$n_{max}$')
    # axs[1].set_xlim(0, 30)
    axs[1].grid(linestyle='--')

    if save:
        plt.savefig('q_nmax30_abs-noabs.pdf', bbox_inches='tight')

def final_plot_normal(save=False):
    Nmax = np.arange(0, 61, 1)
    Q=[]
    for i in aHO_lst:
        Q.append(Q_integral_trapz_aHO_normal(60, 0, i, 1))
    
    plt.figure(figsize=set_size(502.14963))
    for i in range(aHO_lst.shape[0]):
        plt.plot(Nmax, Q[i], 'x-', color=colorFader('#FF00FF', '#000000', i/len(aHO_lst)), markersize=4)
    legend = np.array([['$a_{HO}$=' + str(aHO)] for aHO in aHO_lst]).flatten()
    plt.title('NO ABS')
    plt.legend(legend, loc='lower right', ncol=2)
    plt.ylabel("$Q$ ($e_0$fm$^2$)")
    plt.yticks([0.15, 0.20, 0.25, 0.271, 0.30, 0.40])
    plt.xlabel('$n_{max}$')
    plt.xlim(0, 60)
    plt.grid(linestyle='--')
    
# final_plot(True)

# plot_Q(60, 0, 1)

plt.show()


print(np.mean([Q_integral_trapz_Nmax(150, aHO) for aHO in aHO_lst]), np.std([Q_integral_trapz_Nmax(150, aHO) for aHO in aHO_lst]))
print(np.mean([Q_integral_trapz_Nmax(60, aHO) for aHO in aHO_lst]), np.std([Q_integral_trapz_Nmax(60, aHO) for aHO in aHO_lst]))
print('\n')
print(np.mean([Q_integral_trapz_Nmax_normal(150, aHO) for aHO in aHO_lst]), np.std([Q_integral_trapz_Nmax_normal(150, aHO) for aHO in aHO_lst]))
print(np.mean([Q_integral_trapz_Nmax_normal(60, aHO) for aHO in aHO_lst]), np.std([Q_integral_trapz_Nmax_normal(60, aHO) for aHO in aHO_lst]))
