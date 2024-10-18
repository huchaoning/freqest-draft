from scipy.interpolate import interp1d
import numpy as np

va_curve = interp1d(np.arange(0, 1.6, 0.1), 
                    np.array([0, 
                              0.053, 
                              0.111, 
                              0.154, 
                              0.183, 
                              0.197, 
                              0.202, 
                              0.207, 
                              0.213, 
                              0.220, 
                              0.228, 
                              0.236, 
                              0.244, 
                              0.253, 
                              0.261, 
                              0.269]))
