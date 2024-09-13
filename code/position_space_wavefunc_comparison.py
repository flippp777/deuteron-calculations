import numpy as np
from matplotlib import pyplot as plt
from make_position_space_wavefunc import deuteron_wavefunc


wavefuncs_aHO12_comp = [np.loadtxt('comparative_wave_functions\chi2bM3A_srg0000_NMax\chi2bM3A_srg0000_NMax05_aHO01.20_JLLST10210.txt', skiprows=12, delimiter=',')]
for Nmax in range(10,91,5):
    wavefuncs_aHO12_comp = np.append(wavefuncs_aHO12_comp, 
                                [np.loadtxt(f'comparative_wave_functions\chi2bM3A_srg0000_NMax\chi2bM3A_srg0000_NMax{Nmax}_aHO01.20_JLLST10210.txt', skiprows=12, delimiter=',')], axis=0)

wavefuncs_aHO14_comp = np.array([np.loadtxt('comparative_wave_functions\chi2bM3A_srg0000_NMax\chi2bM3A_srg0000_NMax05_aHO01.40_JLLST10210.txt', skiprows=12, delimiter=',')])
for Nmax in range(10,91,5):
    wavefuncs_aHO14_comp = np.append(wavefuncs_aHO14_comp, 
                                [np.loadtxt(f'comparative_wave_functions\chi2bM3A_srg0000_NMax\chi2bM3A_srg0000_NMax{Nmax}_aHO01.40_JLLST10210.txt', skiprows=12, delimiter=',')], axis=0)

wavefuncs_aHO18_comp= np.array([np.loadtxt('comparative_wave_functions\chi2bM3A_srg0000_NMax\chi2bM3A_srg0000_NMax05_aHO01.80_JLLST10210.txt', skiprows=12, delimiter=',')])
for Nmax in range(10,91,5):
    wavefuncs_aHO18_comp = np.append(wavefuncs_aHO18_comp, 
                                [np.loadtxt(f'comparative_wave_functions\chi2bM3A_srg0000_NMax\chi2bM3A_srg0000_NMax{Nmax}_aHO01.80_JLLST10210.txt', skiprows=12, delimiter=',')], axis=0)

r = wavefuncs_aHO12_comp[0,0]

wavefuncs_aHO12 = [deuteron_wavefunc(5, 1.2, r)[1:3]]
for Nmax in range(10, 91, 5):
    wavefuncs_aHO12 = np.append(wavefuncs_aHO12, [deuteron_wavefunc(Nmax, 1.2, r)[1:3]], axis=0)

# print(np.abs(wavefuncs_aHO12_comp[1,1]))
# print(np.abs(wavefuncs_aHO12[1,0]))
# plt.figure()
# plt.title('L = 0, /')
# plt.plot(r, np.abs(wavefuncs_aHO12[1,0]/wavefuncs_aHO12_comp[1,1]))
# plt.figure()
# plt.title('L = 0, -')
# plt.plot(r, np.abs(wavefuncs_aHO12[1,0]+wavefuncs_aHO12_comp[1,1]))
# plt.figure()
# plt.title('L = 2, /')
# plt.plot(r, np.abs(wavefuncs_aHO12[1,1]/wavefuncs_aHO12_comp[1,2]))
# plt.figure()
# plt.title('L = 2, -')
# plt.plot(r, np.abs(wavefuncs_aHO12[1,1]+wavefuncs_aHO12_comp[1,2]))

wavefuncs_aHO14 = [deuteron_wavefunc(5, 1.4, r)[1:3]]
for Nmax in range(10, 91, 5):
    wavefuncs_aHO14 = np.append(wavefuncs_aHO14, [deuteron_wavefunc(Nmax, 1.4, r)[1:3]], axis=0)


wavefuncs_aHO18 = [deuteron_wavefunc(5, 1.8, r)[1:3]]
for Nmax in range(10, 91, 5):
    wavefuncs_aHO18 = np.append(wavefuncs_aHO18, [deuteron_wavefunc(Nmax, 1.8, r)[1:3]], axis=0)

def integral_approx(r, wavefunc_L0, wavefunc_L2, plot=False):
    dr = r[2]-r[1]
    density = wavefunc_L0**2 + wavefunc_L2**2
    
    L_plot = []
    L = 0
    for i in range(len(r)-1):
        L += min([density[i], density[i+1]]) * dr
        L_plot.append(min([density[i], density[i+1]]))    
    
    U_plot = []
    U = 0
    for i in range(len(r)-1):
        U += max([density[i], density[i+1]]) * dr
        U_plot.append(max([density[i], density[i+1]]))    
    
    if plot:
        plt.plot(r, density)
        plt.plot(np.delete(r,-1), L_plot)
        plt.plot(np.delete(r, -1), U_plot)
        plt.legend(['Density', 'Lower bound', 'Upper bound'])
    return [L * 2, U * 2]


# Normelization approximation for the comparison wave functions
# print('aHO = 1.2')
# for i in range(18):
#     print(integral_approx(r, wavefuncs_aHO12_comp[i, 1], wavefuncs_aHO12_comp[i,2]))
# print('aHO = 1.4')
# for i in range(18):
#     print(integral_approx(r, wavefuncs_aHO14_comp[i, 1], wavefuncs_aHO14_comp[i,2]))
# print('aHO = 1.8')
# for i in range(18):
#     print(integral_approx(r, wavefuncs_aHO18_comp[i, 1], wavefuncs_aHO18_comp[i,2]))


plt.figure()
for i in range(18):
    plt.subplot(3,6,i+1)
    plt.plot(r, np.abs(wavefuncs_aHO12_comp[i,1]))
    plt.plot(r, np.abs(wavefuncs_aHO12[i,0]))
    plt.title(f'aHO = 1.2 fm, L = 0, Nmax = {i*5+5}')
    plt.legend(['Comparison', 'Calculated'])

plt.figure()
for i in range(18):
    plt.subplot(3,6,i+1)
    plt.plot(r, np.abs(wavefuncs_aHO12[i,1]))
    plt.plot(r, np.abs(wavefuncs_aHO12_comp[i,2]))
    plt.title(f'aHO = 1.2 fm, L = 2, Nmax = {i*5+5}')
    plt.legend(['Calculated', 'Comparison'])

# plt.figure()
# for i in range(18):
#     plt.subplot(3,6,i+1)
#     plt.plot(r, np.abs(wavefuncs_aHO14[i,0]))
#     plt.plot(r, np.abs(wavefuncs_aHO14_comp[i,1]))
#     plt.title(f'aHO = 1.4 fm, L = 0, Nmax = {i*5+5}')
#     plt.legend(['Calculated', 'Comparison'])

# plt.figure()
# for i in range(18):
#     plt.subplot(3,6,i+1)
#     plt.plot(r, np.abs(wavefuncs_aHO14[i,1]))
#     plt.plot(r, np.abs(wavefuncs_aHO14_comp[i,2]))
#     plt.title(f'aHO = 1.4 fm, L = 2, Nmax = {i*5+5}')
#     plt.legend(['Calculated', 'Comparison'])

# plt.figure()
# for i in range(18):
#     plt.subplot(3,6,i+1)
#     plt.plot(r, np.abs(wavefuncs_aHO18[i,0]))
#     plt.plot(r, np.abs(wavefuncs_aHO18_comp[i,1]))
#     plt.title(f'aHO = 1.8 fm, L = 0, Nmax = {i*5+5}')
#     plt.legend(['Calculated', 'Comparison'])

# plt.figure()
# for i in range(18):
#     plt.subplot(3,6,i+1)
#     plt.plot(r, np.abs(wavefuncs_aHO18[i,1]))
#     plt.plot(r, np.abs(wavefuncs_aHO18_comp[i,2]))
#     plt.title(f'aHO = 1.8 fm, L = 2, Nmax = {i*5+5}')
#     plt.legend(['Calculated', 'Comparison'])



plt.show()