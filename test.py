import numpy as np
from scipy.special import genlaguerre

def make_f(a):
    def f(x):
        return a * x
    return f
# f(x) = a * x

func_lst = []
for i in range(3):
    func_lst.append(make_f(i))

# func_lst = [0*x, 1*x, 2*x]

coeff = np.arange(3)
# coeff = [0, 1, 2]

func_lst = np.array(func_lst)

def new_func(x):
    new_func = 0
    for i in range(3):
        new_func += func_lst[i](x) * coeff[i]
        print(new_func)
    return new_func
print(new_func(1))

