from math import tau, pi
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.special import erf



def TimeDomainEstimator(estimates_instance):
    from .core import Estimates, qCMOS, SPADE, DI
    c: Estimates = estimates_instance

    origin_shape = c.cropped_data.shape
    flatten = c.cropped_data.reshape(-1, origin_shape[-1])
    works = flatten.shape[0]
    pixels = origin_shape[-1]

    samples = qCMOS.convert2photons(flatten)
    I0, b = c.photons, c.background

    # SPADE measurement
    if c.metadata.measurement.lower() == 'spade':
        _sig = SPADE.SIGMA
        k = qCMOS.convert2photons(c.cropped_data)[..., 0] / qCMOS.convert2photons(c.cropped_data)[..., 1]
        pre_est = 2 * _sig * (1 - np.sqrt(k)) / (1 + np.sqrt(k))

        def _uk(k, theta):
            xi = theta / (2 * _sig)
            return I0/2 * (xi + k)**2 * np.exp(-xi**2) + b

        def _grad_nll(frame):
            def wrapper(theta):
                xi = theta / (2 * _sig)
                k = np.array([-1, 1])
                return - np.sum(frame * np.log(_uk(k, theta)) - _uk(k, theta) - (frame * np.log(frame) - frame)), \
                         np.sum(I0/(2*_sig) * np.exp(-xi**2)*((xi + k) * (xi**2 + k*xi - 1) * (frame/_uk(k, theta) - 1)))
            return wrapper

    # DI
    elif c.metadata.measurement.lower() == 'di':
        _sig = DI.SIGMA  / qCMOS.PIXEL_SIZE
        pre_est = samples @ np.arange(pixels) / samples.sum(-1)

        def _uk(x_, theta):
            z1 = (x_ - theta + 0.5) / (_sig * (2**0.5))
            z2 = (x_ - theta - 0.5) / (_sig * (2**0.5))
            DeltaE = erf(z1)/2 - erf(z2)/2
            return I0 * DeltaE + b

        def _grad_nll(frame):
            x = np.arange(pixels).astype(float)
            def wrapper(theta):
                z1 = (x - theta + 0.5) / (_sig * (2**0.5))
                z2 = (x - theta - 0.5) / (_sig * (2**0.5))
                DeltaE = erf(z1)/2 - erf(z2)/2
                uk = I0 * DeltaE + b
                return - np.sum(frame * np.log(uk) - uk - (frame * np.log(frame) - frame)), \
                       - I0/(_sig*tau**0.5) * np.sum((-np.exp(-z1**2) + np.exp(-z2**2)) * (frame/uk - 1))
            return wrapper
    
    else:
        raise ValueError("measurement must be 'SPADE' or 'DI'")


    # minimize nll
    time_domain = []
    for i in range(works):
        result = minimize(_grad_nll(samples[i]), 
                          x0 = np.ravel(pre_est)[i], 
                          jac = True, 
                          bounds = [(-6*_sig, 6*_sig)], 
                          tol = 1e-8, 
                          options = {'maxls': 100})

        if result.success:
            time_domain.append(result.x.item())
        else:
            raise RuntimeError(f'not converged: {result.message}')


    c.time_domain = np.array(time_domain).reshape(*origin_shape[:-1])

    if c.metadata.measurement.lower() == 'spade':
        c.time_domain = c.time_domain + c.metadata.amplitude

    elif c.metadata.measurement.lower() == 'di':
        c.time_domain = (c.time_domain - pixels/2) * qCMOS.PIXEL_SIZE

    return c
