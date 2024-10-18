import numpy as np
from scipy.interpolate import interp1d
from os import path

pixel_size = 4.6
conversion_factor = 0.107
quantum_efficiency = interp1d(np.linspace(250, 1100, 8501), 
                              np.load(file=path.join(path.dirname(__file__), 'qe.npy')))

def grayscale_value_to_photon_number(grayscale_value, offset=None):
    return (grayscale_value - offset) * conversion_factor / quantum_efficiency

