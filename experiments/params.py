import numpy as np
from fractions import Fraction


__all__ = [
    'Expt',

    'DI',
    'SPADE',

    'qCMOS',
    'DMD'
]


class Expt:
    def validate(params):
        params = np.array(params)
        if (params % 1 == 0).all():
            return (params).astype(int)
        else:
            raise ValueError('invalid params')
        
    # The sigma (AKA characteristic width) of Airy disk is 103um.
    sigma = 103 #um
    expo_time = 0.0451 #s
    # freq_list = np.array([0.1, 0.125, 0.15625, 0.2, 0.22, 0.25, 0.275, 0.3125, 0.34375, 0.4])
    freq_list = np.array([0.1, 0.125, 0.15625, 0.2, 0.22, 0.25, 0.3125, 0.34375, 0.4])

    samples_length = 50
    repeat_times = 200

    sampling_rate = 1 / expo_time

    pic_time = validate(1e6 / (2 * freq_list * sampling_rate))

    samples_pre_cycle = validate([Fraction(str(f)).denominator for f in freq_list])
    video_length = validate((np.ceil(samples_length / samples_pre_cycle) * samples_pre_cycle) * repeat_times)

    cut_point = validate(np.ceil(samples_length / samples_pre_cycle) * samples_pre_cycle)

    spade_avis = [f'spade{i}.avi' for i in range(len(freq_list))]
    di_avis = [f'di{i}.avi' for i in range(len(freq_list))]


class DI:
    ax = 81
    center = 110


class SPADE:
    point1 = (119, 91)
    point2 = (407, 91)


class qCMOS:
    # The camera pixel size is 4.6 um per pixel.
    pixel_size = 4.6 #um


class DMD:
    pixel_size = 19.374725804511403 #um


