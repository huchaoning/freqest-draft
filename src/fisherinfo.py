# from .core import _Share
# import numpy as np
# from math import pi, tau



# class FrequencyCFI:
#     def __init__(self, measurement, N=50, sigma=_Share.SIGMA):
#         self.measurement = measurement

#         self.N = N
#         self.sigma = sigma

#     def _s(self, n, freq, A):
#         da = np.sin(tau * freq * n)
#         db = A * tau * n * np.cos(tau * freq * n)
#         dc = A * np.cos(tau * freq * n)

#         return A * np.sin(tau * freq * n), da, db, dc


#     def cal(self, freq, A, b, pixel_size=qCMOS.PIXEL_SIZE):
#         n = np.arange(self.N)
#         s_list = self._s(n, freq, A)
#         return np.sum(self.measurement.gamma(s_list, *imperfect) * self._ds(n, freq, A) ** 2)


# def FisherInformation(A_list: np.ndarray, freq_list: np.ndarray, sigma=_Share.SIGMA, N=50):
#     results_1, results_2 = [], []
#     n = np.arange(N)
#     for A in A_list:
#         for f in freq_list:
#             _temp = ((tau*n) * np.cos(tau*f*n))**2
#             results_1.append((A/sigma)**2 * _temp.sum())
#         results_2.append(results_1)
#     return np.array(results_2)

# def ApproxFisherInformation(A_list: np.ndarray, sigma=_Share.SIGMA, N=50):
#     n = np.arange(N)
#     results = [2*(A*pi/sigma)**2 * (n**2).sum() for A in A_list]
#     return np.array(results)

# def VelocityFisherInformationMatrix(sampling_rate=20, sigma=_Share.SIGMA, N=10):
#     t = np.arange(N)/sampling_rate
#     fi11 = 1/sigma**2 * np.sum(t**2)
#     fi12 = 1/sigma**2 * np.sum(t**1)
#     fi22 = 1/sigma**2 * np.sum(t**0)
#     return np.array([[fi11, fi12], [fi12, fi22]])
