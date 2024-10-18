# # Import commonly used modules
# from math import *
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm

# from scipy.optimize import curve_fit, minimize

# from matplotlib_inline import backend_inline
# backend_inline.set_matplotlib_formats('svg')
# del backend_inline

# # My personal Python utility module: https://gitee.com/vxyi/myutils.
# try: 
#     from myutils.plotter import *
#     from myutils.macro import push, pull, status
#     set_font('sans-serif', 'normal')
# except ModuleNotFoundError:
#     pass

# from .share import *
# from .bound import *

from .core import *
from .params import *