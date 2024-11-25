import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')
del backend_inline

from matplotlib.font_manager import fontManager
fontManager.addfont('src/lmroman10.otf')
del fontManager

plt.rcdefaults()
plt.rcParams['font.family'] = 'Latin Modern Roman'
plt.rcParams['mathtext.fontset'] = 'cm'


import numpy as np
from tqdm import tqdm
from time import sleep
import os

from datetime import datetime
today = datetime.strftime(datetime.today(), '%Y%m%d')
del datetime


from .core import *
from .estimator import *


# Controller only works on Windows.
import platform
if platform.system() == 'Windows':
    from .controller import *


