#!/usr/bin/env python

import copy
import functools
import numpy as np
import matplotlib.pyplot as plt

import satelliteUtils as satUtil


def momentToEllipse(ixx, iyy, ixy, lo_clip=0.1):

    tmp   = 0.5*(ixx + iyy)
    diff  = ixx - iyy
    tmp2  = np.sqrt(0.25*diff**2 + ixy**2)
    a2    = np.clip(tmp + tmp2, lo_clip, None)
    b2    = np.clip(tmp - tmp2, lo_clip, None)
    ellip = 1.0 - np.sqrt(b2/a2)
    theta = 0.5*np.arctan2(2.0*ixy, diff)

    return ellip, theta, np.sqrt(b2)


    
class MomentManager(object):

    keys = "sumI", "center", "theta", "ellip", "center_perp", "skew", "skew_perp", "b"
    
    def __init__(self, img, kernelWidth, kernelSigma, isCalibration=False):
        self.img          = img
        self.shape        = img.shape
        self.isCal        = isCalibration
        self.std          = img.std()
        self.kernelWidth  = kernelWidth
        self.kernelSigma  = kernelSigma
        self._imageMoment = None
        self._sumI        = None
        self._center      = None
        self._center_perp = None
        self._ellip       = None
        self._theta       = None
        self._skew        = None
        self._skew_perp   = None
        self._b           = None

    def _toEllipse(self):
        if (self._ellip is None):
            ixx, iyy, ixy = self.imageMoment.ixx, self.imageMoment.iyy, self.imageMoment.ixy
            self._ellip, self._theta, self._b   = momentToEllipse(ixx, iyy, ixy)
        
    @property
    def imageMoment(self):
        if self._imageMoment is None:
            kx = np.arange(self.kernelWidth) - self.kernelWidth//2
            self._imageMoment = satUtil.momentConvolve2d(self.img, kx, self.kernelSigma, middleOnly=self.isCal)
        return self._imageMoment
        
    @property
    def sumI(self):
        if self._sumI is None:
            self._sumI = 0.0 if self.isCal else self.img
            #self._sumI = 0.0 if self.isCal else self.imageMoment.i0
            #self._sumI = self.imageMoment.i0
        return self._sumI

    @property
    def center(self):
        if self._center is None:
            ix, iy = self.imageMoment.ix, self.imageMoment.iy
            self._center = np.sqrt(ix*ix + iy*iy)
        return self._center

    @property
    def theta(self):
        if self._theta is None:
            self._toEllipse()
        return self._theta
            
    @property
    def center_perp(self):
        if self._center_perp is None:
            ix, iy      = self.imageMoment.ix, self.imageMoment.iy
            self._center_perp = np.abs(ix*np.sin(self.theta) - iy*np.cos(self.theta))
        return self._center_perp
        
    @property
    def ellip(self):
        if self._ellip is None:
            self._toEllipse()
        return self._ellip

    @property
    def skew(self):
        if self._skew is None:
            ixxx, iyyy = self.imageMoment.ixxx, self.imageMoment.iyyy
            self._skew = np.sqrt(ixxx*ixxx + iyyy*iyyy)
        return self._skew
        
    @property
    def skew_perp(self):
        if self._skew_perp is None:
            ixxx, iyyy     = self.imageMoment.ixxx, self.imageMoment.iyyy
            self._skew_perp= np.abs(ixxx*np.sin(self.theta) - iyyy*np.cos(self.theta))
        return self._skew_perp
        
    @property
    def b(self):
        if self._b is None:
            self._toEllipse()
        return self._b


class MomentLimit(object):
    def __init__(self, name, norm, limitType):
        self.name = name
        self.norm = norm
        self.limitType = limitType

        
class PixelSelector(list):

    def __init__(self, momentManager, calMomentManager):
        super(PixelSelector, self).__init__()
        
        assert(momentManager.kernelWidth == calMomentManager.kernelWidth)
        self.keys = copy.copy(MomentManager.keys)
        
        self.momentManager      = momentManager
        self.calMomentManager   = calMomentManager

    def append(self, limit):
        if limit.name not in self.keys:
            raise ValueError("Limit name must be in:" + str(self.keys))
        limitTypes = ('lower', 'center', 'upper')
        if limit.limitType not in limitTypes:
            raise ValueError("Limit limitType must be in:" + str(limitTypes))
        super(PixelSelector, self).append(limit)

        
    def _norm(self, limit):
        val         = getattr(self.momentManager,    limit.name)
        expectation = getattr(self.calMomentManager, limit.name)
        norm        = (val - expectation)/np.abs(limit.norm)
        return norm

    def _test(self, limit):
        if limit.limitType == 'lower':
            test = self._norm(limit) > 1.0
        elif limit.limitType == 'center':
            test = np.abs(self._norm(limit)) < 1.0
        elif limit.limitType == 'upper':
            test = self._norm(limit) < 1.0
        return test

    def getPixels(self, maxPixels=None):

        keys = getattr(self, 'keys')
        accumulator = np.ones(self.momentManager.shape, dtype=bool)
        for limit in self:
            test = self._test(limit)
            accumulator &= test
            #print key, accumulator.sum(), test.sum()

        if maxPixels:
            # a no-op since we have no way to choose.  We could selected randomly?
            # The parameter can be used by the PValuePixelSelector, which can sort by probability.
            pass
            
        return accumulator


        
        
class PValuePixelSelector(PixelSelector):

    def __init__(self, *args, **kwargs):
        self.thresh = kwargs.get('thresh')
        super(PValuePixelSelector, self).__init__(*args, **kwargs)
    
    def _test(self, limit):
        z     = self._norm(limit)

        # These aren't real probabilities, just functions with properties that go to 1 or 0 as needed.
        # This would be trivial with exp() and log() functions, but they're very expensive,
        # so these approximations use only simple arithmatic.
        
        # Go to 1 for z > 1.  the function is very close to 1 - exp(-x**2)
        # it would go to 1 at large z for both +ve and -ve, so we have to suppress the negative side.
        if limit.limitType == 'lower':
            logp = -0.5/(z*z + 0.0001)
            logp[z < 0.0] = 0.0001
        elif limit.limitType == 'center':
            logp = -0.5*z*z

        # This is the opposite of 'lower'.  I use the same function, but shift it by 2
        # it keeps the values below z~1 and suppresses those above z=2
        elif limit.limitType == 'upper':
            z -= 2
            logp = -0.5/(z*z + 0.0001)
            logp[z > 0.0] = 0.0001
        return logp
        
    def getPixels(self, maxPixels=None):
        n = 0
        logp = np.zeros(self.momentManager.shape)
        for limit in self:
            n += 1
            logp += self._test(limit)

        thresh1 = self.thresh or -0.5*n
        ret = logp > thresh1
        if maxPixels:
            nth = 1.0*maxPixels/logp.size
            if ret.sum() > maxPixels:
                thresh2 = np.percentile(logp, 100*(1.0-nth))
                ret = logp > thresh2
        return ret




        
if __name__ == '__main__':

    n = 256
    kwid = 15
    ksig = 9
    
    img = np.random.normal(size=(n,n))
    img[n/2,n/2] += 1
    mm = MomentManager(img, kernelWidth=kwid, kernelSigma=ksig)

    cal = np.zeros((kwid,kwid))
    cal[kwid/2,kwid/2] += 1
    cmm = MomentManager(cal, kernelWidth=kwid, kernelSigma=ksig)

    limits = { k:1.0 for k in MomentManager.keys }
    limits['sumI'] = -1.0
    limits['theta'] = None
    
    norm = PixelSelector(mm, cmm, limits)
    print norm.ellip[n/2,n/2]
    good = norm.getPixels()
    print good.sum()

    pval = PValuePixelSelector(mm, cmm, limits)
    print pval.ellip[n/2,n/2]
    good = pval.getPixels(thresh=1.0e-10)
    print good.sum()
    
    
