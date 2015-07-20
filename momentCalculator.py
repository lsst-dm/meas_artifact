#!/usr/bin/env python

import copy
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas

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
            self._center = np.sqrt(ix**2 + iy**2)
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
            self._skew = np.sqrt(ixxx**2 + iyyy**2)
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

    def _test(self, limit):
        val         = getattr(self.momentManager,    limit.name)
        expectation = getattr(self.calMomentManager, limit.name)
        norm        = (val - expectation)/np.abs(limit.norm)
        if limit.limitType == 'lower':
            test = norm > 1.0
        elif limit.limitType == 'center':
            test = np.abs(norm) < 1.0
        elif limit.limitType == 'upper':
            test = norm < 1.0
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

        self.cache = {}
        self.done = []
        
    def _test(self, limit):
        if limit.name in self.cache:
            delta, delta2, neg = self.cache[limit.name]
            expectation = getattr(self.calMomentManager, limit.name)
        else:
            val         = getattr(self.momentManager,    limit.name)
            expectation = getattr(self.calMomentManager, limit.name)
            delta       = val - expectation
            neg         = delta < 0.0
            delta2      = delta**2
            self.cache[limit.name] = (delta, delta2, neg)
            
        zz = delta2/(limit.norm**2) + 0.0001
        
        # These aren't real probabilities, just functions with properties that go to 1 or 0 as needed.
        # This would be trivial with exp() and log() functions, but they're very expensive,
        # so these approximations use only simple arithmatic.
        
        # Go to 1 for z > 1.  the function is very close to 1 - exp(-x**2)
        # it would go to 1 at large z for both +ve and -ve, so we have to suppress the negative side.
        if limit.limitType == 'lower':
            neg2logp      = 1.0/zz
            neg2logp[neg] = 1.0/0.0001
            
        elif limit.limitType == 'center':
            neg2logp = zz

        # This is the opposite of 'lower'.  I use the same function, but shift it by 2
        # it keeps the values below z~1 and suppresses those above z=2
        elif limit.limitType == 'upper':
            x = delta/limit.norm
            z = x - 2.0
            zz = z**2 + 0.0001
            pos = x > 2.0
            neg2logp = 1.0/zz
            neg2logp[pos] = 1.0/0.0001
        else:
            raise ValueError("Unknown limit type.")

        return neg2logp
        
    def getPixels(self, maxPixels=None):
        n = 0
        neg2logp = np.zeros(self.momentManager.shape)
        for limit in self:
            n += 1
            val = self._test(limit)
            neg2logp += val
        logp = -0.5*neg2logp

        # logp = -z**2/2 is contributed by each parameter considered
        # So to keep n-sigma points use:
        # 1-sigma:    0.5   # 67% of real candidates accepted
        # 0.7-sigma:  0.25  # 52% accepted
        # 0.5-sigma:  0.125 # 38% accepted
        thresh1 = self.thresh or -0.125*n #-0.5*n
        ret = logp > thresh1
        
        #print "Thresh", thresh1, ret.size, ret.sum()
        if False:
            fig = figure.Figure()
            can = FigCanvas(fig)
            ax = fig.add_subplot(111)
            #ax.hist(qq, bins=200, range=(-2*0.5*n, 0), edgecolor='none')
            dat = self.momentManager.sumI
            ax.plot(dat, logp, 'k.')
            ax.plot(dat[ret], logp[ret], 'r.')
            ax.set_xlim([0, 10.0])
            fig.savefig('phist.png')
            self.done.append(limit.name)

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
    cmm = MomentManager(cal, kernelWidth=kwid, kernelSigma=ksig, isCalibration=True)

    sumI = MomentLimit('sumI', 1.0, 'center')
    
    norm = PixelSelector(mm, cmm)
    norm.append(sumI)
    
    print mm.ellip[n/2,n/2]
    good = norm.getPixels()
    print good.sum()

    pval = PValuePixelSelector(mm, cmm)
    pval.append(sumI)
    
    print mm.ellip[n/2,n/2]
    good = pval.getPixels(maxPixels=1000)
    print good.sum()
    
    
