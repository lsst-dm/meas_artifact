#!/usr/bin/env python

import time
import collections
import numpy as np
from scipy import ndimage as ndimg


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

    
    
def hesseCluster(theta, x, y):
    """
    @theta  angles
    @x     x coordinates
    @y     y coordinates

    return list of (r_i,theta_i)
    """

    dtheta = 0.01

    t1 = time.time()
    
    rdist = 100
    tdist = 0.15
    
    t = theta.copy()

    # clean the thetas
    r = x*np.cos(t) + y*np.sin(t)
    neg, = np.where(r < 0.0)
    t[neg] += np.pi
    cycle, = np.where(t > 2.0*np.pi + 0.2)
    t[cycle] -= 2.0*np.pi
    r = x*np.cos(t) + y*np.sin(t)
    
    print "clean", time.time() - t1
        
    t2 = t + dtheta
    r2 = x*np.cos(t2) + y*np.sin(t2)
    print "cos", time.time() - t1

    xx1, xx2 = np.meshgrid(x, x)
    yy1, yy2 = np.meshgrid(y, y)

    print "aftermesh", time.time() - t1
    
    dx = np.abs(xx1 - xx2)
    dy = np.abs(yy1 - yy2)
    dd = dx + dy

    print "afterabs", time.time() - t1
    
    # this is the theta we get if we just draw a line between points in pixel space
    w0 = np.where(dx == 0)
    dx[w0] = 1.0
    print "where", time.time() - t1
    
    intercept = yy1 - (dy/dx)*xx1
    sign = np.sign(intercept)
    print "atan0", time.time() - t1
    pixel_theta = np.arctan2(dy, dx) + sign*np.pi/2.0
    

    print "atan", time.time() - t1
    
    # convert each point to a line
    m = (r2 - r)/dtheta
    b = r - t*m
    
    # get the distance between points
    tt1, tt2 = np.meshgrid(t, t)
    rr1, rr2 = np.meshgrid(r, r)
    good = (np.abs(tt1 - tt2) < tdist) & (np.abs(rr1 - rr2) < rdist)
    bad  = ~good
    
    # solve for the intersections between all lines
    mm1, mm2 = np.meshgrid(m, m)
    bb1, bb2 = np.meshgrid(b, b)
    trace = np.arange(mm1.shape[0], dtype=int)
    dmm = mm1 - mm2
    dmm[trace,trace] = 1.0
    parallel = np.where(np.abs(dmm) < 0.01)
    dmm[parallel] = 1.0
    tt = (bb2 - bb1)/dmm
    rr = bb1 + mm1*tt
    
    tt[bad] = 0.0
    rr[bad] = 0.0

    w = np.zeros(tt.shape) + 1.0e-7
    
    # weight by the pixel distance (farther is better, less degenerate)
    w[good] += dd[good]

    # de-weight points that have moved us far from where we started
    dtr = np.abs((tt - t)*(rr - r)) + 1.0
    theta_discrepancy = np.abs( tt - pixel_theta ) + 0.01

    w /= dtr*theta_discrepancy

    t_new = np.average(tt, axis=0, weights=w)
    r_new = np.average(rr, axis=0, weights=w)

    
    # use original values for things that didn't converge
    t0 = (t_new < 1.0e-6)
    t_new[t0] = t[t0]
    r0 = (r_new < 1.0e-6)
    r_new[r0] = r[r0]

    print time.time() - t1
    return r_new, t_new, r, x, y


def hesseIter(theta, xx, yy, nIter=3):

    r, t, _r, _xx, _yy = hesseCluster(theta, xx, yy)
    for i in range(nIter):
        r, t, _, _xx, _yy = hesseCluster(t, xx, yy)
    return r, t, _r, _xx, _yy 


def hesseBin(r0, theta0, bins=200, rMax=4096, ncut=4, navg=0.0):

    r = r0
    theta = theta0
    
    theta_margin = 0.4
    
    non_trivial = (np.abs(theta) > 1.0e-2) & (np.abs(r) > 1.0*rMax/bins)
    non_bleed   = np.abs(theta - np.pi/2.0) > 1.0e-2

    ok = non_trivial  & non_bleed

                  
    bin2d, r_edge, t_edge = np.histogram2d(r[ok], theta[ok], bins=(bins,bins),
                                           range=((0.0, rMax), (-theta_margin, theta_margin+2.0*np.pi)) )

    wrpos, wtpos = np.where(bin2d > 1)
    if len(wrpos):
        avgPerBin = np.mean(bin2d[wrpos,wtpos])
    else:
        avgPerBin = 1.0
    thresh = max(ncut, navg*avgPerBin)

    locus, numLocus = ndimg.label(bin2d > thresh, structure=np.ones((3,3)))

    rs, ts, idx, drs, dts = [], [], [], [], []
    for i in range(numLocus):
        loc_r,loc_t = np.where(locus == i + 1)
        
        peak_ti, peak_ri = 0.0, 0.0
        max_val = 0.0
        for i in range(len(loc_t)):
            val = bin2d[loc_r[i],loc_t[i]]
            if val > max_val:
                max_val = val
                peak_ti = loc_t[i]
                peak_ri = loc_r[i]
        

            min_ti = max(peak_ti-1, 0)
            max_ti = min(peak_ti+1, bins-1)
            min_ri = max(peak_ri-1, 0)
            max_ri = min(peak_ri+1, bins-1)

            # wider set
            wmin_ti = max(peak_ti-3, 0)
            wmax_ti = min(peak_ti+3, bins-1)
            wmin_ri = max(peak_ri-3, 0)
            wmax_ri = min(peak_ri+3, bins-1)
            
        tlo, thi = t_edge[min_ti], t_edge[max_ti+1]
        rlo, rhi = r_edge[min_ri], r_edge[max_ri+1]
        # wide edges
        wtlo, wthi = t_edge[wmin_ti], t_edge[wmax_ti+1]
        wrlo, wrhi = r_edge[wmin_ri], r_edge[wmax_ri+1]
        nbox = len(loc_t)
        
        wtmp = np.where((theta >= tlo) & (theta < thi) & (r >= rlo) & (r < rhi))[0]
        t_tmp = np.median(theta[wtmp])
        dt_tmp = theta[wtmp].std()
        r_tmp = np.median(r[wtmp])
        dr_tmp = r[wtmp].std()

        # wide stats
        wwtmp = np.where((theta >= wtlo) & (theta < wthi) & (r >= wrlo) & (r < wrhi))[0]
        wt_tmp = np.median(theta[wwtmp])
        wdt_tmp = theta[wwtmp].std()
        wr_tmp = np.median(r[wwtmp])
        wdr_tmp = r[wwtmp].std()

        
        #print "%6.1f %6.1f  %6.3f %6.3f  %6.1f %6.1f   %6.3f %6.1f  %3d  %3d" % (loc_t.mean(), loc_r.mean(), 0.5*(tlo + thi), thi-tlo, 0.50*(rlo+ rhi), rhi-rlo, t_tmp, r_tmp,  len(wtmp), nbox)

        # don't accept theta < 0 or > 2pi
        if t_tmp < 0.0 or t_tmp > 2.0*np.pi:
            continue

        # if including neighbouring cells increases our stdev, we didn't converge well and the solution
        # is probably bad.  If only r or only theta are broad, then it might be ok ... keep it.
        rgrow = wdr_tmp/dr_tmp
        tgrow = wdt_tmp/dt_tmp
        grow = np.sqrt(rgrow**2 + tgrow**2)
        print r_tmp, t_tmp, rgrow, tgrow, grow

        if rgrow > 2.0 and tgrow > 2.0:
            continue
            
        rs.append(r_tmp)
        drs.append(dr_tmp)
        ts.append(t_tmp)
        dts.append(dt_tmp)
        
        w = np.where((theta0 >= tlo) & (theta0 < thi) & (r0 >= rlo) & (r0 < rhi))[0]
        idx.append(w)

    # check for wrapped-theta doubles,
    # - pick the one with the lowest stdev
    # - this is rare, but a bright near-vertical trail can be detected near theta=0 and theta=2pi
    # --> the real trail is rarely exactly vertical, so one solution will not converge nicely.
    #     ... the stdev of thetas will be wider (10x).
    n = len(rs)
    kill_list = []
    for i in range(n):
        for j in range(i,n):
            dr = abs(rs[i] - rs[j])
            dt = abs(ts[i] - ts[j])
            if dr < 10 and dt > 1.9*np.pi:
                print "d_thetas: ", dts[i], dts[j]
                bad = i if dts[i] > dts[j] else j
                kill_list.append(bad)

    rs_good, ts_good, idx_good = [],[],[]
    for i in range(n):
        if i in kill_list:
            continue
        rs_good.append(rs[i])
        ts_good.append(ts[i])
        idx_good.append(idx[i])
                
    return bin2d, r_edge, t_edge, rs_good, ts_good, idx_good


    
HoughSolution = collections.namedtuple('HoughSolution', 'r theta x y rNew thetaNew binMax')

class HoughSolutionList(list):
    def __init__(self, nSolutions, binMax, r, theta):
        self.nSolutions = nSolutions
        self.binMax     = binMax
        self.r          = r
        self.theta      = theta
    

class HoughTransform(object):
    def __init__(self, bins, thresh, rMax=None, maxPoints=1000, nIter=1):
        self.bins   = bins
        self.thresh = thresh
        self.rMax   = rMax
        
        # things get slow with more than ~1000 points, shuffle and cut
        self.maxPoints = maxPoints
        self.nIter = nIter

        self.overlapRange = 0.2
        
    def __call__(self, thetaIn, xIn, yIn):

        
        rIn, thetaIn = hesseForm(thetaIn, xIn, yIn)

        # wrap the points so we don't have a discontinuity at 0 or 2pi
        theta0, (r0, x0, y0) = twoPiOverlap(thetaIn, (rIn, xIn, yIn), overlapRange=self.overlapRange)
        
        nPoints = len(r0)

        if nPoints == 0:
            return ()
        
        np.random.seed(44)
        r, theta, x, y = r0, theta0, x0, y0
        if nPoints > self.maxPoints:
            idx = np.arange(nPoints, dtype=int)
            np.random.shuffle(idx)
            idx = idx[:self.maxPoints]
            r, theta, x, y = r0[idx], theta0[idx], x0[idx], y0[idx]

            
        # improve the r,theta locations
        rNew, thetaNew, _r, _x, _y = hesseIter(theta, x, y, nIter=self.nIter)

        rMax = self.rMax
        if self.rMax is None:
            rMax = np.sqrt(x.max()**2 + y.max()**2)
                
        # bin the data in r,theta space; get r,theta that pass our threshold as a satellite trail
        bin2d, r_edge, theta_edge, rs, thetas, idx = hesseBin(rNew, thetaNew,
                                                              bins=self.bins, rMax=rMax,
                                                              ncut=self.thresh)
        
        numLocus = len(thetas)
        solutions = HoughSolutionList(numLocus, bin2d.max(), rIn, thetaIn)
        for i in range(numLocus):
            _x, _y = x[idx[i]], y[idx[i]]
            rnew, tnew = rNew[idx[i]], thetaNew[idx[i]]
            solutions.append( HoughSolution(rs[i], thetas[i], _x, _y, rnew, tnew, len(idx[i])) )
        return solutions
        
