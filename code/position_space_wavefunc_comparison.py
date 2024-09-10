import numpy as np
from matplotlib import pyplot as plt
from make_position_wavefunc import deuteron_wavefunc


wavefuncs_aHO12_comp = [np.loadtxt('comparative_wave_functions/chi2b_srg0000_NMax05_aHO01.20_JLLST10210.txt', skiprows=12, delimiter=',')]
for Nmax in range(10,91,5):
    wavefuncs_aHO12_comp = np.append(wavefuncs_aHO12_comp, 
                                [np.loadtxt(f'comparative_wave_functions/chi2b_srg0000_NMax{Nmax}_aHO01.20_JLLST10210.txt', skiprows=12, delimiter=',')], axis=0)

wavefuncs_aHO14_comp = np.array([np.loadtxt('comparative_wave_functions/chi2b_srg0000_NMax05_aHO01.40_JLLST10210.txt', skiprows=12, delimiter=',')])
for Nmax in range(10,91,5):
    wavefuncs_aHO15_comp = np.append(wavefuncs_aHO14_comp, 
                                [np.loadtxt(f'comparative_wave_functions/chi2b_srg0000_NMax{Nmax}_aHO01.40_JLLST10210.txt', skiprows=12, delimiter=',')], axis=0)
    
wavefuncs_aHO18_comp= np.array([np.loadtxt('comparative_wave_functions/chi2b_srg0000_NMax05_aHO01.80_JLLST10210.txt', skiprows=12, delimiter=',')])
for Nmax in range(10,91,5):
    wavefuncs_aHO18_comp = np.append(wavefuncs_aHO18_comp, 
                                [np.loadtxt(f'comparative_wave_functions/chi2b_srg0000_NMax{Nmax}_aHO01.80_JLLST10210.txt', skiprows=12, delimiter=',')], axis=0)

r = wavefuncs_aHO12_comp[0,0,:]

wavefuncs_aHO12 = [deuteron_wavefunc(5, 1.2, r)[1:3]]
for Nmax in range(10, 91, 5):
    wavefuncs_aHO12 = np.append(wavefuncs_aHO12, [deuteron_wavefunc(Nmax, 1.2, r)[1:3]], axis=0)


wavefuncs_aHO14 = [deuteron_wavefunc(5, 1.4, r)[1:3]]
for Nmax in range(10, 91, 5):
    wavefuncs_aHO14 = np.append(wavefuncs_aHO14, [deuteron_wavefunc(Nmax, 1.4, r)[1:3]], axis=0)


wavefuncs_aHO18 = [deuteron_wavefunc(5, 1.8, r)[1:3]]
for Nmax in range(10, 91, 5):
    wavefuncs_aHO18 = np.append(wavefuncs_aHO18, [deuteron_wavefunc(Nmax, 1.8, r)[1:3]], axis=0)

