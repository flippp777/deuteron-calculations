import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import quad

from make_position_space_wavefunc import deuteron_wavefunc



