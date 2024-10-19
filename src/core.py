import os
from math import *
import numpy as np
from datetime import datetime, timedelta, timezone

from .estimator import freq_estimator
from .api import *


__all__ = [
    'qCMOS',
    'DMD',

    'SPADE',
    'DI',
]


#################### 
#    Equipments    #
####################
class qCMOS(Dcam):
    # The camera pixel size is 4.6 um per pixel.
    PIXEL_SIZE = 4.6 #um
    CONVERSION_FACTOR = 0.107

    
    def __init__(self, iDevice=0):
        super().__init__(iDevice)
        self.ez_isopen = False


    def __enter__(self):
        Dcamapi.init()
        self.dev_open()
        self.prop_setvalue(DCAM_IDPROP.SENSORCOOLER, DCAMPROP.SENSORCOOLER.MAX)
        self.ez_isopen = True
        return self


    def __exit__(self, *args):
        if self.ez_isopen:
            self.cap_stop()
            self.buf_release()
            self.dev_close()
            Dcamapi.uninit()
            self.ez_isopen = False
            print('exited')


    def ez_exposure_time(self, exposure_time):
        self.prop_setvalue(DCAM_IDPROP.EXPOSURETIME, exposure_time)


    def ez_triggersource_masterpluse(self, burst_times, interval):
        self.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE, DCAMPROP.TRIGGERSOURCE.MASTERPULSE)
        self.prop_setvalue(DCAM_IDPROP.MASTERPULSE_MODE, DCAMPROP.MASTERPULSE_MODE.BURST)
        self.prop_setvalue(DCAM_IDPROP.MASTERPULSE_TRIGGERSOURCE, DCAMPROP.MASTERPULSE_TRIGGERSOURCE.SOFTWARE)

        self.prop_setvalue(DCAM_IDPROP.MASTERPULSE_BURSTTIMES, burst_times)
        self.prop_setvalue(DCAM_IDPROP.MASTERPULSE_INTERVAL, interval)

    
    def ez_triggersource_external(self):
        self.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE, DCAMPROP.TRIGGERSOURCE.EXTERNAL)


    def ez_temperature(self):
        return self.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE)
    

    def ez_roi(self, X0, Y0, W, H):
        self.prop_setvalue(DCAM_IDPROP.SUBARRAYHPOS, X0)
        self.prop_setvalue(DCAM_IDPROP.SUBARRAYVPOS, Y0)
        self.prop_setvalue(DCAM_IDPROP.SUBARRAYHSIZE, W)
        self.prop_setvalue(DCAM_IDPROP.SUBARRAYVSIZE, H)
        self.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE,  2)


    def ez_wait_capture(self, timeout=18446744073709551616):
        while True:
            if self.wait_event(DCAMWAIT_CAPEVENT.CYCLEEND, timeout) is not False:
                break

    
    def ez_read_buf(self, iFrame, read_timestamp=True):
        frame = self.buf_getframe(iFrame)
        timestamp = self.ez_fmt_time(frame[0]) if read_timestamp else 0
        data = frame[1]
        return data, timestamp


    @classmethod
    def ez_fmt_time(self, buf_frame: DCAMBUF_FRAME):
        total_seconds = buf_frame.timestamp.sec + buf_frame.timestamp.microsec / 1_000_000
        china_time = datetime.fromtimestamp(total_seconds, tz=timezone.utc).astimezone(timezone(timedelta(hours=8)))
        return china_time.strftime('%Y-%m-%d %H:%M:%S.%f') 




class DMD(ALP4):
    PIXEL_SIZE = 19.374725804511403 #um

    TRIANGLE_SEQ = np.ravel((np.array([np.arange(-5, 6), np.arange(-5, 6)])).T)[::-1][1:-1]
    TRIANGLE_SEQ = np.concatenate([TRIANGLE_SEQ, TRIANGLE_SEQ[::-1]])
    
    def __init__(self, version='4.3', libDir=os.path.join(f'{os.path.dirname(__file__)}', 'api/')):
        super().__init__(version, libDir)
        self.ez_isopen = False


    def __enter__(self):
        self.Initialize()
        self.ez_load_seq([self.ez_single_pixel(0)])
        self.Run(loop=False)
        self.Wait()
        self.ez_isopen = True
        return self


    def __exit__(self, *args):
        self.Halt()
        try:
            self.FreeSeq()
        except ValueError:
            pass
        self.Free()
        self.ez_isopen = False
        print('exited')

    
    def ez_single_pixel(self, pixels):
        img = np.ones([self.nSizeY, self.nSizeX]) * (2**8 - 1)
        img[self.nSizeY//2 - pixels, self.nSizeX//2 + pixels] = 0
        return img.ravel()


    def ez_load_seq(self, Imgs, PictureTime=50):
        imgSeq = np.concatenate(Imgs)
        self.SeqAlloc(nbImg=len(Imgs), bitDepth=1)
        self.SeqPut(imgData=imgSeq)
        self.SeqControl(ALP_BIN_MODE, ALP_BIN_UNINTERRUPTED)
        self.SetTiming(pictureTime=PictureTime)





######################
#    Measurements    #
######################
class _Share:
    @classmethod
    def load(cls, file):
        cls.file = file


    @classmethod
    def read(cls):
        cls.raw = np.load(cls.file).astype(float)


    @classmethod
    def freq_est(cls):
        cls.freq = np.array([freq_estimator(sample) for sample in cls.td])



class SPADE(_Share):
    X_AXIS = 89
    POINT_1 = 407
    POINT_2 = 119

    ROI = {'X0': 2128, 'Y0': 720, 'W': 180, 'H': 500}

    @classmethod
    def crop(cls):
        cls.cropped = cls.raw[..., (cls.POINT_1, cls.POINT_2), cls.X_AXIS]


    @classmethod
    def photons(cls):
        cls.pn = cls.cropped.sum(-1) - 400


    @classmethod
    def td_est(cls):
        cls.td = cls.cropped[..., 0] - cls.cropped[..., 1]


class DI(_Share):
    pass
