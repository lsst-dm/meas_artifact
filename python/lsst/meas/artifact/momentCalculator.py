#!/usr/bin/env python

import copy
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas

import satelliteUtils as satUtil


def momentToEllipse(ixx, iyy, ixy, loClip=0.1):
    """Convert moments to ellipse parameters (numpy-safe)

    @param ixx     2nd moment x
    @param iyy     2nd moment y
    @param ixy     2nd moment xy
    @param loClip  Minium value to accept for either A (semi-major) or B (semi-minor)

    @return ellip, theta, B    Ellipticity 1-B/A, Pos.Angle theta, and semi-minor axis B
    """

    tmp   = 0.5*(ixx + iyy)
    diff  = ixx - iyy
    tmp2  = np.sqrt(0.25*diff**2 + ixy**2)
    a2    = np.clip(tmp + tmp2, loClip, None)
    b2    = np.clip(tmp - tmp2, loClip, None)
    ellip = 1.0 - np.sqrt(b2/a2)
    theta = 0.5*np.arctan2(2.0*ixy, diff)

    return ellip, theta, np.sqrt(b2)


    
class MomentManager(object):
    """Handle calculation of moments for all pixels in an image.

    We'll try to do this in an on-demand way, so we only calculate something
    if it's being used.
    """
    
    
    keys = "sumI", "center", "theta", "ellip", "center_perp", "skew", "skew_perp", "b"
    
    def __init__(self, img, kernelWidth, kernelSigma, isCalibration=False):
        """Construct

        @param img            The image with moments we want computed.
        @param kernelWidth    The kernel width to use in pixels
        @param kernelSigma    Gaussian sigma for a weight function applied multiplicatively to the kernel
        @param isCalibration  Is this a calibration image?
                                (If so, don't convolve, just get the calib pixel [the center])

        """
        self.img          = img
        self.shape        = img.shape
        self.isCal        = isCalibration
        self.std          = img.std()
        self.kernelWidth  = kernelWidth
        self.kernelSigma  = kernelSigma

        # properties
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
        """Compute the convolutions"""
        if self._imageMoment is None:
            kx = np.arange(self.kernelWidth) - self.kernelWidth//2
            self._imageMoment = satUtil.momentConvolve2d(self.img, kx, self.kernelSigma,
                                                         middleOnly=self.isCal)
        return self._imageMoment
        
    @property
    def sumI(self):
        """Get the sum of pixel values"""
        if self._sumI is None:
            self._sumI = 0.0 if self.isCal else self.img
        return self._sumI

    @property
    def center(self):
        """Get the centroid offset (1st moment)"""
        if self._center is None:
            ix, iy = self.imageMoment.ix, self.imageMoment.iy
            self._center = np.sqrt(ix**2 + iy**2)
        return self._center

    @property
    def theta(self):
        """Get the position angle w.r.t. the x-axis in radians."""
        if self._theta is None:
            self._toEllipse()
        return self._theta
            
    @property
    def center_perp(self):
        """Get the centroid offset (1st moment) perpendicular to the alignment."""
        if self._center_perp is None:
            ix, iy      = self.imageMoment.ix, self.imageMoment.iy
            self._center_perp = np.abs(ix*np.sin(self.theta) - iy*np.cos(self.theta))
        return self._center_perp
        
    @property
    def ellip(self):
        """Get the ellipticity: e = 1 - B/A """
        if self._ellip is None:
            self._toEllipse()
        return self._ellip

    @property
    def skew(self):
        """Get the skewness (3rd moment)"""
        if self._skew is None:
            ixxx, iyyy = self.imageMoment.ixxx, self.imageMoment.iyyy
            self._skew = np.sqrt(ixxx**2 + iyyy**2)
        return self._skew
        
    @property
    def skew_perp(self):
        """Get the skewness (3rd moment) perpendicular to the alignment."""
        if self._skew_perp is None:
            ixxx, iyyy     = self.imageMoment.ixxx, self.imageMoment.iyyy
            self._skew_perp= np.abs(ixxx*np.sin(self.theta) - iyyy*np.cos(self.theta))
        return self._skew_perp
        
    @property
    def b(self):
        """Get the 'semi-minor axis', B"""
        if self._b is None:
            self._toEllipse()
        return self._b


class MomentLimit(object):
    """A light-weight container for info about limits.
    """
    def __init__(self, name, value, limitType):
        """Construct.

        @param name       Name of the limit (must be in MomentManager.keys)
        @param value      The value to use as a limit.
        @param limitType  Is the limit 'lower', 'center', or 'upper' limit?
        """
        self.name      = name
        self.value     = value
        self.limitType = limitType

        
class PixelSelector(list):
    """A simple pixel selector.

    Inherit from a list, and we'll contain a list of our MomentLimit objects.
    We'll go through our list, and any pixels which are numerically within the limits
    for all MomentLimit objects "pass" and are kept.

    In the end, we return a boolean image with True set for accepted pixels.
    """
    
    def __init__(self, momentManager, calMomentManager):
        """Construct

        @param momentManager    MomentManager for the image we're selecting from
        @param calMomentManager The MomentManager for the calibration image.
        """
        
        super(PixelSelector, self).__init__()
        
        assert(momentManager.kernelWidth == calMomentManager.kernelWidth)
        self.keys = copy.copy(MomentManager.keys)
        
        self.momentManager      = momentManager
        self.calMomentManager   = calMomentManager

    def append(self, limit):
        """Overload our parent list's append so that we can verify the MomemntList being appended

        If all is well, we'll call our parent's append()
        
        @param limit   The MomentLimit object being added to this selector.
        """
        if limit.name not in self.keys:
            raise ValueError("Limit name must be in:" + str(self.keys))
        limitTypes = ('lower', 'center', 'upper')
        if limit.limitType not in limitTypes:
            raise ValueError("Limit limitType must be in:" + str(limitTypes))
        super(PixelSelector, self).append(limit)

        
    def _test(self, limit):
        """Helper method to determine pass/fail for a specified MomentLimit

        @param limit   The MomentLimit to test.

        @return test   The result image of the test (passing pixels are True)
        """
        
        val         = getattr(self.momentManager,    limit.name)
        expectation = getattr(self.calMomentManager, limit.name)
        norm        = (val - expectation)/np.abs(limit.value)
        if limit.limitType == 'lower':
            test = norm > 1.0
        elif limit.limitType == 'center':
            test = np.abs(norm) < 1.0
        elif limit.limitType == 'upper':
            test = norm < 1.0
        return test

    def getPixels(self, maxPixels=None):
        """Check against all MomentLimit and return an image with pixels which passed all tests.

        @param maxPixels   Limit the number of pixels
                           (not implemented here as there's no obvious way to sort them)
        """
        keys = getattr(self, 'keys')
        accumulator = np.ones(self.momentManager.shape, dtype=bool)
        for limit in self:
            test = self._test(limit)
            accumulator &= test

        if maxPixels:
            # a no-op since we have no way to choose.  We could selected randomly?
            # The parameter can be used by the PValuePixelSelector, which can sort by probability.
            pass
            
        return accumulator


        
        
class PValuePixelSelector(PixelSelector):
    """A P-Value based pixel selector.

    This serves the same purpose as the PixelSelector, but computes a p-value for each pixel.
    The MomentLimits are used as 1-sigma thresholds, and the resulting sum of log(p) values
    is computed.  Pixels meeting a specified threshold are kept.
    """
    
    def __init__(self, *args, **kwargs):
        """Construct.
        """
        self.thresh = kwargs.get('thresh')
        super(PValuePixelSelector, self).__init__(*args, **kwargs)

        # cache some value in case the user wants to test the same thing twice
        # e.g. an upper limit and a lower limit.
        self.cache = {}
        self.done = []
        
    def _test(self, limit):
        """Test the significance of pixels for this MomentLimit.

        @param limit  The MomentLimit to test.
        """
        
        if limit.name in self.cache:
            delta, delta2, neg = self.cache[limit.name]
            expectation = getattr(self.calMomentManager, limit.name)
        else:
            val         = getattr(self.momentManager,    limit.name)
            expectation = getattr(self.calMomentManager, limit.name)
            delta       = val - expectation
            neg         = delta <= 0.0
            delta2      = delta**2
            self.cache[limit.name] = (delta, delta2, neg)

        # If z is normalized, our Gaussian is P = exp(-z**2/2)
        # Or ... z**2 = -2*log(P)

        divByZeroValue = 0.001
        
        zz = delta2/(limit.value**2) + divByZeroValue
        
        # Some of these aren't real probabilities, just functions that go to 1 or 0 as needed.
        # This would be trivial with exp() and log() functions, but they're very expensive,
        # so these approximations use only simple arithmatic.
        
        # Go to 1 for z > 1.  the function is very close to 1 - exp(-x**2)
        # it would go to 1 at large z for both +ve and -ve, so we have to suppress the negative side.
        if limit.limitType == 'lower':
            neg2logp      = 1.0/zz
            neg2logp[neg] = 1.0/divByZeroValue
            
        elif limit.limitType == 'center':
            neg2logp = zz

        # This is the opposite of 'lower'.
        # it keeps the values below z~1 and suppresses those above z=2
        elif limit.limitType == 'upper':
            neg2logp = zz
            neg2logp[neg] = divByZeroValue

        else:
            raise ValueError("Unknown limit type.")

        return neg2logp
        
    def getPixels(self, maxPixels=None):
        """Get the pixels which pass all MomentLimit tests.

        @param maxPixels   Return no more than this many 'pass' pixels.  (Sorting by p-value)
        """
        
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

        if maxPixels:
            nth = 1.0*maxPixels/logp.size
            if ret.sum() > maxPixels:
                thresh2 = np.percentile(logp, 100*(1.0-nth))
                ret = logp > thresh2
        return ret


