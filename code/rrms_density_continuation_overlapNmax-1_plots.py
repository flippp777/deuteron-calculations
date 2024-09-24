import numpy as np
import matplotlib.pyplot as plt

from rrms_position_probability_density_continuation_overlapNmax_minus_1 import find_cont_par_cut, deut_density, log_model, rrms_integral_trapz

aHO_lst = np.array([1.2, 1.4, 1.6, 1.8, 2.0]) 
Nmax_lst = np.arange(0, 62, 2)
r = np.linspace(0, 100, 1000)


def test_plots(r, NNmax, dNmax):
    Nmax_lst = np.arange(3, NNmax + 1, dNmax)
    
    for j in range(aHO_lst.size):
        rrms = []
        plt.figure()
        plt.subplot(121)

        rrms_lst = []
        rrms_mean_lst = []
        for Nmax in Nmax_lst:
            slope , y_sect, connect, rrms, norm, fit_points = find_cont_par_cut(Nmax, aHO_lst[j])
            r_cont = np.array([connect, 20])
            plt.plot(r, deut_density(r, Nmax, aHO_lst[j]) * norm, 'b')
            plt.plot(r_cont, np.exp(log_model(r_cont, slope, y_sect)) * norm, "r")
            # plt.scatter(fit_points[0], np.exp(fit_points[1]))
            plt.semilogy()
            plt.grid(linestyle='--')
            rrms_lst.append(rrms)
        
        plt.title("Wellenfunktionen und exponentielle Fortzetzung für aHO = " + str(aHO_lst[j]) + " fm (logaryhtmische y-Axhse)") 
        plt.ylim(10**(-6), 1)
        plt.xlim(0, 20)
        plt.legend(["Wellenfunktion", "Fortsetzung"])
        
        rmax, dr = max(r), r[1]-r[0]
        rrms_full_wave = rrms_integral_trapz(60, aHO_lst[j], 2, rmax, dr)
        plt.subplot(122)
        plt.plot(rrms_full_wave[0], rrms_full_wave[1], 'x-')
        plt.plot(Nmax_lst, rrms_lst, 'x-')
        plt.title('$R_{rms}$ bei $a_{HO} = $' + str(aHO_lst[j]) + ' fm aus Wellenfunktion und exponentieller Fortsetzung')
        plt.legend(["Wellenfunktion", 'Wellenfunktion + Fortsetzung'])
        plt.xlabel('$N_{max}$')
        plt.ylabel("$R_{rms}$ (fm)")
        plt.yticks([1.2, 1.4, 1.6, 1.8, 1.96998, 2.0, 2.2])
        plt.grid(linestyle='--')



            
test_plots(r, 61, 2)


plt.show()