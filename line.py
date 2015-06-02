#!/usr/bin/env python

import sys, os, re, copy
import numpy
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas

def lineFit(x, y, dy=None):
    """A standard linear least squares line fitter with errors and chi2."""
    
    N = len(x)
    if N < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    no_err = False
    if dy is None:
        no_err = True
        dy = numpy.ones(N)
        
    var = dy**2
    S  = (1.0/var).sum()
    Sx = (x/var).sum()
    Sy = (y/var).sum()
    Sxx = ((x**2)/var).sum()
    Sxy = ((x*y)/var).sum()
    Delta = S*Sxx - Sx**2

    if (no_err) or Delta < 1.0e-6:
        Stt = ((x - Sx/S)**2).sum()
        Sty = ((x - Sx/S)*y).sum()
    
    if no_err or Delta < 1.0e-6:
        bb = (1.0/Stt)*Sty
        aa = (Sy - Sx*bb)/S
        var_aa = (1.0/S)*(1.0 + Sx**2/(S*Stt))
        var_bb = (1.0/Stt)
    else:
        bb = ( S*Sxy - Sx*Sy ) / Delta
        aa = ( Sxx*Sy - Sx*Sxy ) / Delta
        var_aa = Sxx/Delta
        var_bb = S / Delta

    rms_aa = numpy.sqrt(numpy.abs(var_aa))
    rms_bb = numpy.sqrt(numpy.abs(var_bb))

    # coefficient of correlation
    if no_err or Delta < 1.0e-6:
        cov_ab = (-Sx/(S*Stt))
        r_ab   = (cov_ab/(rms_aa*rms_bb))
    else:
        cov_ab = (-Sx/Delta)
        r_ab   = -Sx/numpy.sqrt(S*Sxx)

    # get chi_squared
    X2 = (((y - aa - bb*x) / dy)**2).sum()

    return aa, rms_aa, bb, rms_bb, r_ab, X2
  


def robustPolyFit(x, y, order, nbin=3, sigma=3.0, niter=1):

    xNew, yNew = copy.copy(x), copy.copy(y)
    
    # bin and take medians in each bin
    epsilon = 1.0e-6
    xmin, xmax = xNew.min() - epsilon, xNew.max() + epsilon

    step = (xmax-xmin)/(nbin)
    xMeds, yMeds, yErrs = [], [], []
    for i in range(nbin):
        w = numpy.where( (xNew > xmin + i*step) & (xNew <= xmin + (i+1)*step) )
        if len(xNew[w]) == 0:
            continue
        xMeds.append(numpy.median(xNew[w]))
        yMeds.append(numpy.median(yNew[w]))
        yErr = numpy.std(yNew[w])/numpy.sqrt(len(yNew[w]))
        yErrs.append(yErr)
        
    # use these new coords to fit the line ... with sigma clipping if requested
    xNew, yNew, dyNew = numpy.array(xMeds), numpy.array(yMeds), numpy.array(yErrs)

    # if there's only one point in a bin, dy=0 ... use the average error instead
    w0 = numpy.where(dyNew == 0)[0]
    wnot0 = numpy.where(dyNew > 0)[0]
    if len(w0) > 0:
        if len(wnot0) > 0:
            meanError = numpy.mean(dyNew[wnot0])
        # if *all* bins have a single point, then all dy are zero
        # take a blind guess and use stdev of all values we got
        else:
            meanError = numpy.std(yNew)

        # last ... if we have errors of zero, use 1.0
        # sounds silly, but if we fit a line to a difference value
        # and it's the difference of identical values
        # (eg. instFlux and modFlux are loaded with the same value)
        # then the dy could be zero
        if meanError == 0.0:
            meanError = 1.0
        dyNew[w0] = meanError
    
    for i in range(niter):

        #ret = numpy.polyfit(xNew, yNew, order)
        #p = ret[0:2]
        a, da, b, db, rab, x2 = lineFit(xNew, yNew, dyNew)
        
        residuals = yNew - numpy.polyval((a, b), xNew)

        if i == 0:
            mean = numpy.median(residuals)
        else:
            mean = numpy.mean(residuals)
            
        std = numpy.std(residuals)

        if niter > 1:
            w = numpy.where( (numpy.abs(residuals - mean)/std) < sigma )
            xNew = xNew[w]
            yNew = yNew[w]
            dyNew = dyNew[w]

    # return b, db, a, da
    return a, da, b, db


def intersect(a, da, b, db):

    a1 = a+da
    a2 = a-da
    b1 = b-db
    b2 = b+db

    x = (a2 - a1) / (b1 - b2)
    y = a1 + b1*x
    return x, y


def butterfly(x_in, y_in):

    if False:
        xm = x_in.mean()
        xs = x_in.std()
        ym = y_in.mean()
        ys = y_in.std()
        x = (x_in - xm)/xs
        y = (y_in - ym)/ys
    else:
        xm = 0.0
        xs = 1.0
        ym = 0.0
        ys = 1.0
        x  = x_in
        y  = y_in
    ix = []
    iy = []
    for xx, yy, switch in ((x,y,False), (y,x,True)):
        if False:
            fit = lineFit(xx, yy)
        else:
            fit = robustPolyFit(xx, yy, 1, nbin=5, niter=2)
        aa, daa, bb, dbb = fit[0:4]
        ixx, iyy = intersect(aa, daa, bb, dbb)
        if switch:
            ixx, iyy = iyy, ixx
        ix.append(xs*ixx + xm)
        iy.append(ys*iyy + ym)
    return numpy.mean(ix), numpy.mean(iy)
    
    
if __name__ == '__main__':

    a = 1.0
    b = 200.0
    n = 100
    ran = 1.0
    sigma = 50.0
    x = ran*numpy.random.uniform(size=n)
    noise = sigma*numpy.random.normal(size=n)
    y = a + b*x + noise

    aa, daa, bb, dbb, r, chi = lineFit(x, y, dy=sigma*numpy.ones(n))

    print aa, daa, bb, dbb, r, chi
    xx = ran*numpy.arange(n)/n


    ####################3
    ddaa = 10.0*daa
    ddbb = 10.0*dbb
    x2 = numpy.array([])
    y2 = numpy.array([])
    for i in range(-10,10):
        a = aa - i*ddaa/10
        b = bb + i*ddbb/10
        xtmp = ran*numpy.random.uniform(size=20)
        x2 = numpy.append(x2, xtmp)
        y2 = numpy.append(y2, a + b*xtmp + 5.0*numpy.random.normal(size=20))

    dy2 = 50.0*numpy.ones(len(x2))

    
    aa2, daa2, bb2, dbb2, r2, chi2 = lineFit(x2, y2, dy=dy2)
    ix2, iy2 = intersect(aa2, daa2, bb2, dbb2)
    print aa2, daa2, bb2, dbb2, r2, chi2, ix2, iy2
    aa3, daa3, bb3, dbb3, r3, chi3 = lineFit(y2, x2, dy=dy2)
    iy3, ix3 = intersect(aa3, daa3, bb3, dbb3)
    print aa3, daa3, bb3, dbb3, r3, chi3, ix3, iy3

    ixx, iyy = butterfly(x2, y2)
    
    fig = figure.Figure()
    canvas = FigCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'r.')
    ax.plot(xx, aa+bb*xx, 'g-')
    ax.plot(xx, aa-daa + (bb+dbb)*xx, 'b-')
    ax.plot(xx, aa+daa + (bb-dbb)*xx, 'b-')

    ax.plot(x2, y2, 'c.')
    ax.plot(x2, aa2+bb2*x2, 'k-')
    ax.plot(x2, aa2-1*daa2 + (bb2+1*dbb2)*x2, 'm-')
    ax.plot(x2, aa2+1*daa2 + (bb2-1*dbb2)*x2, 'm-')

    ax.scatter([ix2, ix3], [iy2, iy3], c='r', s=150, marker='o', facecolor='none', edgecolor='r')
    ax.scatter([ixx], [iyy], c='g', s=250, marker='o', facecolor='none', edgecolor='g')

    
    fig.savefig("line.png")
    

    
