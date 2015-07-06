#!/usr/bin/env python

import copy
import functools
import numpy as np
import matplotlib.pyplot as plt

import satellite_utils as satUtil


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


                
        
class PixelSelector(object):

    def __init__(self, momentManager, calMomentManager, momentLimits):
        
        assert(momentManager.kernelWidth == calMomentManager.kernelWidth)
        self.keys = copy.copy(MomentManager.keys)
        
        self.momentManager      = momentManager
        self.calMomentManager   = calMomentManager
        self.momentLimits       = {}

        for attr in self.keys:
            self.momentLimits[attr] = momentLimits.get(attr, None)
            setattr(self, attr, functools.partial(self._norm, attr))

    def __getattribute__(self, attr):
        keys   = super(PixelSelector, self).__getattribute__('keys')
        method = super(PixelSelector, self).__getattribute__(attr)
        if attr in keys:
            return method()
        else:
            return method
            
    def _norm(self, name):
        limit       = self.momentLimits[name]
        if limit is None:
            return np.zeros(self.momentManager.shape)
        else:
            mid = self.momentManager.kernelWidth//2
            val         = getattr(self.momentManager, name)
            expectation = getattr(self.calMomentManager, name)
            #print name, expectation, limit
            return (val - expectation)/np.abs(limit)

    def _test(self, name):
        limit       = self.momentLimits[name]
        if limit and limit < 0.0:
            test = np.abs(getattr(self, name)) > 1.0
        else:
            test = np.abs(getattr(self, name)) < 1.0
        return test

    def getPixels(self, binOnTheta=False, maxPixels=None):

        keys = getattr(self, 'keys')
        accumulator = np.ones(self.momentManager.shape, dtype=bool)
        for key in keys:
            test = self._test(key)
            accumulator &= test
            #print key, accumulator.sum(), test.sum()

        if binOnTheta:
            t = getattr(self.momentManager, 'theta')
            hist, edges = np.histogram(t[accumulator], bins=40)
            best = np.argmax(hist)
            tbest = 0.5*(edges[best] + edges[best+1])
            accumulator &= (np.abs(tbest - t) < 0.2)
            
        if maxPixels:
            # a no-op since we have no way to choose.  We could selected randomly?
            # The parameter can be used by the PValuePixelSelector, which can sort by probability.
            pass
            
        return accumulator


        
        
class PValuePixelSelector(PixelSelector):

    def __init__(self, *args, **kwargs):
        self.thresh = kwargs.get('thresh')
        super(PValuePixelSelector, self).__init__(*args, **kwargs)
    
    def _test(self, name):
        limit = self.momentLimits[name]
        z     = getattr(self, name)
        if limit and limit < 0.0:
            # the real answer is np.log(1.0001 - np.exp(-0.5*z*z)), but is very slow
            # This approx is actually really close 
            return -0.5/(z*z + 0.0001)
        else:
            return -0.5*z*z

    def getPixels(self, binOnTheta=False, maxPixels=None):
        keys = getattr(self, 'keys')
        logp = self._test(keys[0])
        n = 0
        for key in keys:
            if key in self.momentLimits:
                n += 1
            logp += self._test(key)

        if binOnTheta:
            t = getattr(self.momentManager, 'theta')
            tProb = t[logp > -0.5*n]
            hist, edges = np.histogram(tProb, bins=120)

            #print len(tProb)
            #plt.plot(edges[:-1], hist, 'r-')
            #plt.savefig("junk.png")

            best = np.argmax(hist)
            tbest = 0.5*(edges[best] + edges[best+1])
            z = (t - tbest)/0.1
            logp += -0.5*z*z
            n += 1

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
    
    
