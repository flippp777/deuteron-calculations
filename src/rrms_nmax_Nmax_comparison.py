import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import trapezoid
import sys

if sys.platform == 'win32':
    sys.path.insert(1, r'C:\Users\phili.DESKTOP\OneDrive\Uni OneDrive\Bachelor\Bachelorthesis\deuteron-calculations\src')
else:
    sys.path.insert(1, '/home/philipp/bachelorthesis/deuteron-calculations/src')

from make.position_space_wavefunc import deuteron_wavefunc
from plotstyle.colorFader import colorFader
from plotstyle.set_size import set_size

plt.style.use('src\stylesheet.mplstyle')

def deut_density(r, Nmax, aHO):
    return deuteron_wavefunc(Nmax, aHO, r)[3]

def rsq_integral_trapz(Nmax, aHO, rmax, dr):
    r_size = int(np.round(rmax / dr))
    r = np.linspace(0, rmax + 1, r_size)
    integr = deut_density(r, Nmax, aHO) * r**4
    return trapezoid(integr, r, dr) / 4      # Warum Faktor 4?


def rrms_integral_trapz(Nmax, Nmin, aHO, Nstep, rmax, dr):
    rrms = []
    Nmax_lst = np.arange(Nmin, Nmax + 1, Nstep)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rsq_integral_trapz, Nmax, aHO, rmax, dr) for Nmax in Nmax_lst]
        for future in futures:
            rrms.append(future.result())
    
    rrms = np.sqrt(np.array(rrms))
    
    return np.array([Nmax_lst, rrms])


aho_lst = [1.1, 1.2, 1.3, 1.4]
nmax_lst = np.arange(1,11,1)

rrms_nmax = []
for aho in aho_lst:
    rrms_nmax.append(rrms_integral_trapz(10, 1, aho, 1, 150, 0.01))
rrms_nmax = np.array(rrms_nmax)

rrms_Nmax = np.array([
    [1.161056, 1.232539, 1.373228, 1.399997, 1.482726, 1.512896, 1.567623, 1.607370, 1.641652, 1.673522],
    [1.236529, 1.322089, 1.454896, 1.473087, 1.577656, 1.578479, 1.642089, 1.664974, 1.703608, 1.732694],
    [1.313415, 1.420203, 1.533740, 1.545947, 1.665010, 1.648953, 1.721657, 1.713929, 1.763416, 1.777995],
    [1.392616, 1.525526, 1.614778, 1.626553, 1.740034, 1.716378, 1.806013, 1.769064, 1.824986, 1.812697]])

plt.figure(figsize=set_size(502.14963))

for i in range(len(aho_lst)):
    plt.plot(rrms_nmax[i,0], rrms_nmax[i,1], 'x-', color=colorFader('#FF00FF', '#000000', i/4))
    plt.plot(nmax_lst, rrms_Nmax[i], 'o--', color=colorFader('#FF00FF', '#000000', i/4))

# for i in range(len(aho_lst)):
#     plt.plot(nmax_lst, rrms_Nmax[i], 'o--', color=colorFader('#FF00FF', '#000000', i/4))

plt.grid(linestyle='--')
plt.xlabel('$\\Tilde{N}_\\text{max}\\approx N_\\text{max}/2$')
plt.ylabel('$R_\\text{rms}$ [fm]')
plt.legend(np.array([['$a_\\text{HO}=$'+str(aho)+' fm, $R_\\text{rms}(\\Tilde{N}_\\text{max})$', '$a_\\text{HO}=$'+str(aho)+' fm, $R_\\text{rms}(N_\\text{max})$'] for aho in aho_lst]).flatten(), ncol=2)

plt.savefig("rrms_nmax_Nmax_comparison.pdf", bbox_inches='tight')

plt.show()


values = np.array([nmax_lst, rrms_nmax[1,1], rrms_Nmax[1], np.abs((rrms_nmax[1,1]-rrms_Nmax[1])/rrms_Nmax[1])*100, 
                    rrms_nmax[2,1], rrms_Nmax[2], np.abs((rrms_nmax[2,1]-rrms_Nmax[2])/rrms_Nmax[2])*100])
# np.savetxt('nmax_Nmax_comparison.txt', values.transpose())
