#!/usr/bin/env python

import copy
import time
import collections
import numpy as np
from scipy import ndimage as ndimg

import matplotlib.pyplot as plt

import hesse_cluster as hesse

def hesseForm(theta_in, x, y):
    """Convert theta, x, y   to Hesse normal form

    @param theta_in    Local position angle in radians (-pi/2 < theta_in < pi/2)
    @param x           Local x coordinate in pixels
    @param y           Local y coordinate in pixels

    @return r, theta   Hesse normal form:  r = x*cos(theta) + y*sin(theta)

    The theta_in we are given is a local position angle wrt to the x-axis.  For an elongated
    shape at position x,y; theta_in is aligned along the *long* axis.  However, the theta
    used in the Hesse normal form is the angle of the vector normal to the local long axis.
    Theta is therefore different from theta_in by +/- pi/2.  The sign is determined by the sign
    of the y-intercept.
    
    The basic geometry is shown at e.g. https://en.wikipedia.org/wiki/Hesse_normal_form
    """
    
    # if the intercept is y > 0, then add pi/2; otherwise subtract pi/2
    intercept = y - x*np.tan(theta_in)
    pihalves  = np.sign(intercept)*0.5*np.pi
    theta     = theta_in + pihalves
    
    # now ... -pi < theta < pi ...  convert to 0..2pi range
    theta[(theta < 0.0)] += 2.0*np.pi
    r         = x*np.cos(theta) + y*np.sin(theta)
    
    return r, theta
        

def twoPiOverlap(theta_in, arrays=None, overlapRange=0.2):
    """
    Take any thetas near zero and *copy* them to above 2*pi.
    Take any thetas near 2*pi and *copy* them to near 0

    Sometimes the theta we're looking for is near 0 or 2*pi and we'd like
    a continue sample of points.  Otherwise, we can end-up with the
    same theta yielding two solutions.
    """
    
    w_0,   = np.where( theta_in < overlapRange )
    w_2pi, = np.where( 2.0*np.pi - theta_in < overlapRange )

    theta = np.append(theta_in, theta_in[w_0] + 2.0*np.pi)
    theta = np.append(theta,    theta_in[w_2pi] - 2.0*np.pi)

    # if we're given other arrays, append in the same way
    outArray = []
    if arrays:
        for arr in arrays:
            tmp = np.append(arr, arr[w_0])
            tmp = np.append(tmp, arr[w_2pi])
            outArray.append(tmp)
            
    return theta, outArray



def thetaAlignment(theta, x, y, limit=3, tolerance=0.19):
    """A helper function to cull the candidate points.

    @param theta      ndarray of thetas
    @param x          ndarray of x pixel coordinates
    @param y          ndarray of y pixel coordinates

    The basic idea here is that for any pair of points, each has a local measure of theta
    and also vector connecting the two points, which also defines a third theta.
    All of these should agree, so we can eliminate any candidate points for which
    either local theta is more than 'tolerance' different from that defined by the
    dx,dy pixel coordinates.

    This only gets you so far.  With ~1000 candidate points, each one will have ~10 chance
    alignments for a reasonable tolerance.  Though the local theta values are uncertain at the +/-0.1
    level, the pixel coordinate-based thetas are much more precise.  So from those ~10, we can
    search for pairs which have pixelTheta values which line up 'too well'.

    The final step is choosing an delta-angle to define as the smallest separation we'd expect
    to see between any two points.  If we assume the points are uniformly distributed, the probability
    1 point will we be found within 'delta' of another is e^(-2*phi) where fill-factor phi=n*delta/range.
    This might look familiar as the Poisson prob for zero events with rate u=-2*phi:
    
       P(x=k) = u^k/k! e^(-u)   -->  P(x=0) = e^(-u)
    
    What we're saying is that if we sit on one of n points in a region 'range',
    the probability of observing 0 points within 'delta' of our position is:
    
       P(x=0) = e^(-2*n*delta/range)
    
    The factor of 2 arises because a point may preceed or follow.
    """
    
    n = len(theta)

    dydx      = np.subtract.outer(y, y)/(np.subtract.outer(x, x) + 0.01)
    # we could call arctan(dydx) here but it's very expensive.
    # We faster if we instead convert our tolerance to test dydx directly.
    tanTheta  = np.tan(theta)
    dTanTheta = tolerance*(1.0 + tanTheta**2)
    aligned1  = np.abs((dydx - tanTheta)/dTanTheta) < 1.0
    aligned2  = np.abs((dydx.transpose() - tanTheta).transpose()/dTanTheta) < 1.0
    aligned   = aligned1 & aligned2

    nNearNeighbours = np.zeros(n)
    newThetas = copy.copy(theta)
    
    # Using variable names  pCloseNeighbour = e^(2*nCand*closeNeighbourTolerance/tolerance)
    pZeroCloseNeighbour     = 0.95
    # compute the closeNeighbourTolerance for which we expect 1 collision
    phi                     = -0.5*np.log(pZeroCloseNeighbour)

    nCand = aligned.sum(axis=1)
    closeNeighbourTolerance = phi*tolerance/nCand

    cut = max(limit, 2)
    w, = np.where(nCand > cut)
    for i in w:
        pixelTheta = np.arctan(dydx[i,aligned[i,:]])
        idx        = np.argsort(pixelTheta)
        diff       = np.diff(pixelTheta[idx])
        didx       = (diff < closeNeighbourTolerance[i])
        
        # how many collisions do we actually have?
        nNearNeighbours[i]      = didx.sum()

        if nNearNeighbours[i] >= limit:
            newThetas[i] = pixelTheta[idx[didx]].mean()
            
    isCandidate = nNearNeighbours >= limit

    if False:
        nAligned    = (aligned1 & aligned2).sum(axis=1)
        isAligned   = nAligned >= limit
        
        fig, ax = plt.subplots(nrows=2, ncols=3)
        ax[0,0].hist(theta, bins=500)
        ax[0,1].hist(np.arctan(dydx).ravel(), bins=500)
        ax[1,0].hist(nAligned, bins=40)
        ax[1,1].hist(theta[isAligned], bins=50, edgecolor='none')
        ax[1,1].hist(theta[isCandidate], bins=50, edgecolor='none', facecolor='r')
        ax[1,2].hist(diffs, bins=200)
        ax[1,2].set_xlim([0.0, 0.01])
        fig.savefig("hist.png")

    return isCandidate, newThetas

    
def improveCluster(theta, x, y):
    """
    @theta  angles
    @x      x pixel coordinates
    @y      y pixel coordinates

    return list of (r_i,theta_i)

    Due to noise in the original image, the theta values are never quite aligned
    with the satellite trail.  The conversion to the normal form r = x*cos(t) + y*sin(t)
    shows that for a given x,y pixel; errors in theta are sinusoids in r(theta).
    So a cluster in r,theta often has streaks of points passing through the true r,theta
    of the satellite trail.  Since we know x,y and the form of r(theta) very well, we
    can compute dr/dtheta = -x*sin(t) + y*cos(t) for each point in r,theta space.  This is a linear
    approximation to r(theta) near the point.  The idea behind this improvement strategy
    is that points coming from the same (real) satellite trail will share a common point
    in r,theta (the premise of the Hough Transform).  For one point, we only know it lies on
    the r(theta) curve, but for two ... the intersection of the curves is a better estimate of r,theta
    for each contributing point.
    """

    dtheta = 0.01

    rdist = 100
    tdist = 0.15
    
    t = theta.copy()
    r = x*np.cos(t) + y*np.sin(t)
    #neg       = (r < -0.2)
    #t[neg]   += np.pi
    #cycle     = (t > 2.0*np.pi + 0.2)
    #t[cycle] -= 2.0*np.pi

    dx = np.subtract.outer(x, x)
    dy = np.subtract.outer(y, y)
    dd = np.sqrt(dx**2 + dy**2)

    w0     = (dx == 0)
    dx[w0] = 1.0

    # this is the theta we get if we just draw a line between points in pixel space
    # We could just use this, but (I'm not sure why) it doesn't do as nice a job.
    intercept   = y - (dy/dx)*x
    sign        = np.sign(intercept)
    pixel_theta = np.arctan(dy/dx) + sign*np.pi/2.0
    pixel_theta[(pixel_theta < 0.0)] += np.pi
    
    # convert each point to a line in r,theta space
    # drdt is slope and 'b' is the intercept
    drdt = -x*np.sin(t) + y*np.cos(t)
    b    = r - t*drdt
    
    # get the distance between points in r,theta space "good" pairs are close together
    isGood = (np.abs(np.subtract.outer(t, t)) < tdist) & (np.abs(np.subtract.outer(r, r)) < rdist)
    isBad  = ~isGood
    
    # solve for the intersections between all lines in r,theta space
    dm           = np.subtract.outer(drdt, drdt).transpose()
    parallel     = (np.abs(dm) < 0.01)
    dm[parallel] = 1.0
    tt           = np.subtract.outer(b, b)/dm
    rr           = b + tt*drdt
    
    tt[isBad] = 0.0
    rr[isBad] = 0.0

    # Create a weight vector to use in a weighted-average
    w = np.zeros(tt.shape) + 1.0e-7

    # weight by the pixel distance (farther is better, less degenerate)
    w[isGood] += dd[isGood]

    # de-weight points that have moved us far from where they started
    dtr               = np.abs((tt - t)*(rr - r)) + 1.0
    # de-weight points where the solved theta disagrees with delta-x,delta-y in pixel space
    theta_discrepancy = np.abs(tt - pixel_theta) + 0.01

    w[isGood] /= (dtr*theta_discrepancy)[isGood]

    # for each point, the 'best' theta is the weighted-mean of all places it's line intersects others.
    t_new = np.average(tt, axis=0, weights=w)
    r_new = x*np.cos(t_new) + y*np.sin(t_new)
    
    # use original values for things that didn't converge
    t0 = (t_new < 1.0e-6)
    t_new[t0] = t[t0]
    r0 = (r_new < 1.0e-6)
    r_new[r0] = r[r0]

    return r_new, t_new, r, x, y



def hesseBin(r0, theta0, bins=200, rMax=4096, thresh=40):
    """Bin r,theta values to find clusters above a threshold

    @param r0         List of r values
    @param theta0     List of theta values
    @param bins       Number of bins to use in each of r,theta (i.e. total bins*bins will be used)
    @param rMax       Specify the range in r
    @param thresh     The count limit in a bin which indicates a detected cluster of points.

    @return bin2d, rEdge, tEdge, rsGood, tsGood, idxGood

    bin2d    the 2D binned data
    rEdge    the edges of the bins in r
    tEdge    the edges of the bins in theta
    rsGood   List of best r values for any loci
    tsGood   List of best theta values for any loci
    idxGood  List of indices (from the input arrays) for any points contributing to any loci
    
    In principal, this function is simple ... bin-up the values of r,theta and see
    if any of the bins have more than 'thresh' points.  Take the input r,thetas which landed
    in any such bin and use them to get the 'best' value of r,theta ... mean, median, whatever.
    However, we also have the check the neighbouring bins, and we need to do
    some basic due diligence to assess whether we believe the locus of points is a real
    satellite trail.  A real trail should converge with almost all points in a 3x3 cell locus, so
    the r,theta RMS of the locus in 3x3 cell and 5x5 cell regions should be about the same.

    """

    
    r     = r0.copy()
    theta = theta0.copy()
    
    overlapRange = 0.2

    # eliminate any underdesirable r,theta values
    notTrivial = (np.abs(theta) > 1.0e-2) & (np.abs(r) > 1.0*rMax/bins)

    # there actually *are* near vertical trails.  Disable this for now.
    #notBleed   = np.abs(theta - np.pi/2.0) > 1.0e-2
    isOk       = notTrivial # & notBleed

    bin2d, rEdge, tEdge = np.histogram2d(r[isOk], theta[isOk], bins=(bins,bins),
                                           range=((0.0, rMax), (-overlapRange, overlapRange+2.0*np.pi)) )
    locus, numLocus = ndimg.label(bin2d > thresh, structure=np.ones((3,3)))

    rs, ts, idx, drs, dts = [], [], [], [], []
    for i in range(numLocus):
        label = i + 1
        loc_r,loc_t = np.where(locus == label)
        
        iThetaPeak, iRPeak = 0.0, 0.0
        maxVal = 0.0
        for i in range(len(loc_t)):
            val = bin2d[loc_r[i],loc_t[i]]
            if val > maxVal:
                maxVal    = val
                iThetaPeak = loc_t[i]
                iRPeak     = loc_r[i]
        

            # iThetaPeak,iRPeak  is the peak count for this label in bin2d

            # get the indices for a 3x3 box with iThetaPeak,iRPeak at the center
            iThetaMin = max(iThetaPeak - 1, 0)
            iThetaMax = min(iThetaPeak + 1, bins - 1)
            iRMin     = max(iRPeak - 1,     0)
            iRMax     = min(iRPeak + 1,     bins - 1)

            # get indices for a (wider) 5x5 box
            iThetaMinWide = max(iThetaPeak - 3, 0)
            iThetaMaxWide = min(iThetaPeak + 3, bins - 1)
            iRMinWide     = max(iRPeak - 3,     0)
            iRMaxWide     = min(iRPeak + 3,     bins - 1)

        tlo, thi = tEdge[iThetaMin], tEdge[iThetaMax + 1]
        rlo, rhi = rEdge[iRMin],     rEdge[iRMax + 1]
        # wide edges
        tloWide, thiWide = tEdge[iThetaMinWide], tEdge[iThetaMaxWide + 1]
        rloWide, rhiWide = rEdge[iRMinWide],     rEdge[iRMaxWide + 1]
        nbox = len(loc_t)

        # for this locus, use the median r,theta for points within the 3x3 box around the peak
        centeredOnPeak = (theta >= tlo) & (theta < thi) & (r >= rlo) & (r < rhi)
        tTmp          = np.median(theta[centeredOnPeak])
        dtTmp         = theta[centeredOnPeak].std()
        rTmp          = np.median(r[centeredOnPeak])
        drTmp         = r[centeredOnPeak].std()

        # also get stats for points within the 5x5 box
        # If there are many points in the 5x5 that aren't in the 3x3, the solution didn't converge
        centeredOnPeakWide = (theta >= tloWide) & (theta < thiWide) & (r >= rloWide) & (r < rhiWide)
        tTmpWide           = np.median(theta[centeredOnPeakWide])
        dtTmpWide          = theta[centeredOnPeakWide].std()
        rTmpWide           = np.median(r[centeredOnPeakWide])
        drTmpWide          = r[centeredOnPeakWide].std()

        # don't accept theta < 0 or > 2pi
        if tTmp < 0.0 or tTmp > 2.0*np.pi:
            continue

        # if including neighbouring cells increases our stdev, we didn't converge well and the solution
        # is probably bad.  If only r or only theta are broad, then it might be ok ... keep it.
        rgrow = drTmpWide/drTmp
        tgrow = dtTmpWide/dtTmp
        grow = np.sqrt(rgrow**2 + tgrow**2)

        if rgrow > 3.0 and tgrow > 3.0:
            print "Warning: Discarding locus at %.1f,%.3f due to poor convergence (%.2f/%.2f, %.2f/%2.f)." % \
                (rTmp, tTmp, drTmp, rgrow, dtTmp, tgrow)
            #continue

        rs.append(rTmp)
        drs.append(drTmp)
        ts.append(tTmp)
        dts.append(dtTmp)

        w = (theta0 >= tlo) & (theta0 < thi) & (r0 >= rlo) & (r0 < rhi)
        idx.append(w)

        
    # check for wrapped-theta doubles,
    # - pick the one with the lowest stdev
    # - this is rare, but a bright near-vertical trail can be detected near theta=0 *and* theta=2pi
    # --> the real trail is rarely exactly vertical, so one solution will not converge nicely.
    #     ... the stdev of thetas will be wider by a factor of ~10x
    n = len(rs)
    kill_list = []
    for i in range(n):
        for j in range(i,n):
            dr = abs(rs[i] - rs[j])
            dt = abs(ts[i] - ts[j])
            if dr < 10 and dt > 1.9*np.pi:
                bad = i if dts[i] > dts[j] else j
                kill_list.append(bad)

    rsGood, tsGood, idxGood = [],[],[]
    for i in range(n):
        if i in kill_list:
            continue
        rsGood.append(rs[i])
        tsGood.append(ts[i])
        idxGood.append(idx[i])
                
    return bin2d, rEdge, tEdge, rsGood, tsGood, idxGood


HoughSolution = collections.namedtuple('HoughSolution', 'r theta x y rNew thetaNew binMax resid')
"""A container for results from a HoughSolution"""

Residual = collections.namedtuple('Residual', 'med iqr')
"""A container for stats on the residuals of a HoughSolution"""

class HoughSolutionList(list):
    """A list of clusters found in a Hough Transform

    This is used as a return value for a HoughTransform function object.
    It inherits from a Python list, but includes a few attributes
    relevant to the overall Hough solution.
    """

    def __init__(self, binMax, r, theta):
        """Construct for a HoughSolutionList

        @param binMax   The highest count level found in any bin.
        @param r        The r values determined as part of the transform
        @param theta    The theta values input to the houghTransform
        """
        self.binMax     = binMax
        self.r          = r
        self.theta      = theta
    

class HoughTransform(object):
    """Compute a Hough Transform for a set of x,y pixel coordinates with an good estimate of theta

    This class is really just a housing for a Hough Transform.  The parameters needed are
    provided at construction and a __call__ method is defined to do the work.
    """
    
    def __init__(self, bins, thresh, rMax=None, maxPoints=1000, nIter=1, maxResid=3.0):
        """Construct a HoughTransform object.

        @param bins           Number of bins to use in each of r,theta (i.e. total bins*bins will be used)
        @param thresh         The count limit in a bin which indicates a detected cluster of points.
        @param rMax           Specify the range in r
        @param maxPoints      Maximum number of points to allow (solution gets slow for >> 1000)
        @param nIter          No. of times to iterate the solution (more than ~1-3 rarely makes a difference)
        @param maxResid       The max coordinate residual (pixels) to accept a solution (currently an IQR).

        fitResid requires explanation.  The final hurdle in assessing a solution is to compute residuals
        for any points we have which contributed to the final r,theta.  For a good solution, all points
        should lie close to the line defined by r,theta.  This is currently implemented as an
        inter-quartile-range of the residuals.  It should be a few pixels.  If half the points are off by
        this much, it's probably not a great solution.  Other options might have been a 'max' or an 'rms',
        but the IQR is robust against bad solutions while still forgiving of few bad points.
        """
        
        self.bins   = bins
        self.thresh = thresh
        self.rMax   = rMax
        
        # things get slow with more than ~1000 points, shuffle and cut
        self.maxPoints = maxPoints
        self.nIter = nIter

        self.maxResid = maxResid
        # You shouldn't need to touch this.  We allow this much slack around theta = 0 or 2*pi
        # Points within overlapRange of 0 or 2pi are copied to wrap.  The allows good
        # solutions to be found very near theta = 0 and theta=2pi
        self.overlapRange = 0.2

        
    def __call__(self, thetaIn, xIn, yIn):
        """Compute the Hough Transform

        @param  thetaIn        The local theta values at pixel locations (should be within 0.2 of true value)
        @param  xIn            The x pixel coordinates corresponding to thetaIn values
        @param  yIn            the y pixel coordinates corresponding to thetaIn values

        @return solutions      A HoughSolutionList with an entry for each locus found.
        """
        
        rIn, thetaIn = hesseForm(thetaIn, xIn, yIn)

        # wrap the points so we don't have a discontinuity at 0 or 2pi
        theta0, (r0, x0, y0) = twoPiOverlap(thetaIn, (rIn, xIn, yIn), overlapRange=self.overlapRange)
        
        nPoints = len(r0)

        if nPoints == 0:
            return HoughSolutionList(0, rIn, thetaIn)
        
        np.random.seed(44)
        r, theta, x, y = r0, theta0, x0, y0
        if nPoints > self.maxPoints:
            idx = np.arange(nPoints, dtype=int)
            np.random.shuffle(idx)
            idx = idx[:self.maxPoints]
            r, theta, x, y = r0[idx], theta0[idx], x0[idx], y0[idx]


        # improve the r,theta locations
        rNew, thetaNew, _r, _x, _y = improveCluster(theta, x, y)
        for i in range(self.nIter):
            rNew, thetaNew, _r, _x, _y = improveCluster(thetaNew, x, y)

        rMax = self.rMax
        if self.rMax is None:
            rMax = np.sqrt(x.max()**2 + y.max()**2)
                
        # bin the data in r,theta space; get r,theta that pass our threshold as a satellite trail
        bin2d, rEdge, thetaEdge, rs, thetas, idx = hesseBin(rNew, thetaNew, thresh=self.thresh,
                                                            bins=self.bins, rMax=rMax)
        
        numLocus = len(thetas)
        solutions = HoughSolutionList(bin2d.max(), rIn, thetaIn)
        for i in range(numLocus):
            _x, _y = x[idx[i]], y[idx[i]]
            rnew, tnew = rNew[idx[i]], thetaNew[idx[i]]
            residual = _x*np.cos(thetas[i]) + _y*np.sin(thetas[i]) - rs[i]

            med = np.percentile(residual, 50.0)
            q1  = np.percentile(residual, 25.0)
            q3  = np.percentile(residual, 75.0)
            iqr = q3 - q1
            n   = idx[i].sum()

            # see if there's a significant 2nd-order term in a polynomial fit.
            order2limit = 1.0e-3
            poly = np.polyfit(np.arange(n), residual, 2)
            if False:
                fig, ax = plt.subplots(nrows=1, ncols=1)
                xTmp = np.arange(n)
                ax.plot(xTmp, residual, 'r.')
                ax.plot(xTmp, np.poly1d(poly)(xTmp), 'b-')
                fig.savefig("poly.png")

            isReallyActuallyLinear = (iqr < self.maxResid) & (np.abs(poly[0]) < order2limit)
            
            if isReallyActuallyLinear:
                
                residStat = Residual(med, q3 - q1)
                solution  = HoughSolution(rs[i], thetas[i], _x, _y, rnew, tnew, n, residStat)
                solutions.append(solution)

            else:
                print "WARNING: Rejecting solution: r=%.1f,theta=%.3f  " \
                    "(IQR=%.2f [limit=%.2f]  2nd-order coeff = %.2g [limit=%.2g])" % \
                    (rs[i], thetas[i], iqr, self.maxResid, poly[0], order2limit)

        return solutions
        
