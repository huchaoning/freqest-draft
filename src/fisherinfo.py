from .core import _Share
import numpy as np
from math import pi, tau



class CFIM:
    def __init__(self) -> None:
        pass



def FisherInformation(A_list: np.ndarray, freq_list: np.ndarray, sigma=_Share.SIGMA, N=50):
    results_1, results_2 = [], []
    n = np.arange(N)
    for A in A_list:
        for f in freq_list:
            _temp = ((tau*n) * np.cos(tau*f*n))**2
            results_1.append((A/sigma)**2 * _temp.sum())
        results_2.append(results_1)
    return np.array(results_2)

def ApproxFisherInformation(A_list: np.ndarray, sigma=_Share.SIGMA, N=50):
    n = np.arange(N)
    results = [2*(A*pi/sigma)**2 * (n**2).sum() for A in A_list]
    return np.array(results)

def VelocityFisherInformationMatrix(sampling_rate=20, sigma=_Share.SIGMA, N=10):
    t = np.arange(N)/sampling_rate
    fi11 = 1/sigma**2 * np.sum(t**2)
    fi12 = 1/sigma**2 * np.sum(t**1)
    fi22 = 1/sigma**2 * np.sum(t**0)
    return np.array([[fi11, fi12], [fi12, fi22]])
