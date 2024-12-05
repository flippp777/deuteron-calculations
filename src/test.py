import numpy as np
import matplotlib.pyplot as plt


Nmax_lst = np.arange(1, 11)
aHO_lst = [1.1, 1.2, 1.3]
NNmax_lst = np.arange(4, 11)

rrms_lsts = []
rrms_fullwav = []
for aHO in aHO_lst:
    rrms_lst = []
    for Nmax in Nmax_lst:
        rrms_lst.append(Nmax)
    rrms_lsts.append(rrms_lst)
    
rrms_lsts = np.array(rrms_lsts)

mean_lst = []
std_lst = []
for i in NNmax_lst:
    mean_lst.append(rrms_lsts[:, i-4:i])

for mean in mean_lst:
    print(mean)
    print("\n")