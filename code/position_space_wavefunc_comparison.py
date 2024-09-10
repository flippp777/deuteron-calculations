import numpy as np
from matplotlib import pyplot as plt
from make_position_wavefunc import deuteron_wavefunc


wavefuncs_aHO12 = np.array([np.loadtxt('comparative_wave_functions/chi2b_srg0000_NMax05_aHO01.20_JLLST10210.txt', skiprows=12, delimiter=',')])
for Nmax in range(10,91,5):
    wavefuncs_aHO12 = np.append(wavefuncs_aHO12, 
                                [np.loadtxt(f'comparative_wave_functions/chi2b_srg0000_NMax{Nmax}_aHO01.20_JLLST10210.txt', skiprows=12, delimiter=',')], axis=0)

wavefuncs_aHO14 = np.array([np.loadtxt('comparative_wave_functions/chi2b_srg0000_NMax05_aHO01.40_JLLST10210.txt', skiprows=12, delimiter=',')])
for Nmax in range(10,91,5):
    wavefuncs_aHO14 = np.append(wavefuncs_aHO14, 
                                [np.loadtxt(f'comparative_wave_functions/chi2b_srg0000_NMax{Nmax}_aHO01.40_JLLST10210.txt', skiprows=12, delimiter=',')], axis=0)
    
wavefuncs_aHO18 = np.array([np.loadtxt('comparative_wave_functions/chi2b_srg0000_NMax05_aHO01.80_JLLST10210.txt', skiprows=12, delimiter=',')])
for Nmax in range(10,91,5):
    wavefuncs_aHO18 = np.append(wavefuncs_aHO18, 
                                [np.loadtxt(f'comparative_wave_functions/chi2b_srg0000_NMax{Nmax}_aHO01.80_JLLST10210.txt', skiprows=12, delimiter=',')], axis=0)
