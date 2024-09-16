import numpy as np

for i in [1.2, 1.4, 1.8]:
    with open(f"comparative_wave_functions\chi2bM3A_srg0000_NMax\chi2bM3A_srg0000_NMax05_aHO0{i}0_JLLST10210.evp", "rt") as f:
        data = f.read()
        data_n = data.replace('rList->    { ','')
        data_n = data_n.replace('phi0List-> {','')
        data_n = data_n.replace('phi1List-> {','')
        data_n = data_n.replace('}', '')
        data_n = data_n.replace(', \n', '\n')
        
    with open(f'comparative_wave_functions\chi2bM3A_srg0000_NMax\chi2bM3A_srg0000_NMax05_aHO0{i}0_JLLST10210.txt', "w") as f:
        f.write(data_n)

for j in range(10, 91, 5):
    for i in [1.2, 1.4, 1.8]:
        with open(f"comparative_wave_functions\chi2bM3A_srg0000_NMax\chi2bM3A_srg0000_NMax{j}_aHO0{i}0_JLLST10210.evp", "rt") as f:
            data = f.read()
            data_n = data.replace('rList->    { ','')
            data_n = data_n.replace('phi0List-> {','')
            data_n = data_n.replace('phi1List-> {','')
            data_n = data_n.replace('}', '')
            data_n = data_n.replace(', \n', '\n')
            
        with open(f'comparative_wave_functions\chi2bM3A_srg0000_NMax\chi2bM3A_srg0000_NMax{j}_aHO0{i}0_JLLST10210.txt', "w") as f:
            f.write(data_n)