#!/usr/bin/env python


import numpy as np
from scipy import ndimage as ndimg

import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas

import line

def hesse_cluster(theta, x, y):
    """
    @theta  angles
    @x     x coordinates
    @y     y coordinates

    return list of (r_i,theta_i)
    """

    dtheta = 0.01
    
    rdist = 100
    tdist = 0.15
    
    t = theta.copy()
    t2 = t + dtheta
    
    # clean the thetas
    r = x*np.cos(t) + y*np.sin(t)
    neg, = np.where(r < 0.0)
    t[neg] += np.pi
    cycle, = np.where(t > 2.0*np.pi)
    theta[cycle] -= 2.0*np.pi
    r = x*np.cos(t) + y*np.sin(t)
    r2 = x*np.cos(t2) + y*np.sin(t2)

    xx1, xx2 = np.meshgrid(x, x)
    yy1, yy2 = np.meshgrid(y, y)
    dd = np.sqrt( (xx1 - xx2)**2 + (yy1 - yy2)**2 )
    
    # convert each point to a line
    m = (r2 - r)/(t2 - t)
    b = (t*r2 - t2*r)/(t - t2)
    
    # get the distance between points
    tt1, tt2 = np.meshgrid(t, t)
    rr1, rr2 = np.meshgrid(r, r)
    dt = np.abs(tt1 - tt2)
    dr = np.abs(rr1 - rr2)

    # solve for the intersections between all lines
    mm1, mm2 = np.meshgrid(m, m)
    bb1, bb2 = np.meshgrid(b, b)
    tt = (bb2 - bb1)/(mm1 - mm2)
    rr = bb1 + mm1*(bb2 - bb1)/(mm1 - mm2)

    # replace each theta with the mean of the intersection-thetas
    wr,wt = np.where(
        (~np.isnan(tt)) & (~np.isnan(rr)) &
        (np.isfinite(tt)) & (np.isfinite(rr)) &
        (dt < tdist) & (dr < rdist)
    )
    # there must be a smart way to get the complement of the wr,wt indices ... this isn't it.
    nr, nt = np.where(
        (np.isnan(tt)) | (np.isnan(rr)) |
        (~np.isfinite(tt)) | (~np.isfinite(rr)) |
        (dt > tdist) | (dr > rdist)
    )

    tt[nr,nt] = 0.0
    rr[nr,nt] = 0.0

    w = np.zeros(tt.shape)+ 1.0e-7
    w[wr,wt] += 1.0

    # weight by the pixel distance (farther is better, less degenerate)
    w *= dd

    # de-weight points that have moved us far from where we started
    dtt = np.abs(tt - theta)
    drr = np.abs(rr - r)
    w /= dtt*drr + 1.0
    
    t_new = np.average(tt, axis=0, weights=w)
    r_new = np.average(rr, axis=0, weights=w)

    # use original values for things that didn't converge
    t0 = np.where(np.abs(t_new) < 1.0e-6)
    t_new[t0] = theta[t0]
    r0 = np.where(np.abs(r_new) < 1.0e-6)
    r_new[r0] = r[r0]
    
    return r_new, t_new, r, x, y


def hesse_iter(theta, xx, yy, niter=3):

    r, t, _r, _xx, _yy = hesse_cluster(theta, xx, yy)
    for i in range(niter):
        r, t, _, _xx, _yy = hesse_cluster(t, xx, yy)
    return r, t, _r, _xx, _yy  #t.mean(), r.mean()


def hesse_bin(r, theta, bins=200, r_max=4096, ncut=4, navg=1.0):

    
    non_trivial, = np.where( (np.abs(theta) > 1.0e-3) | (np.abs(r) > 1.0e-3))
    
    bin2d, r_edge, t_edge = np.histogram2d(r[non_trivial], theta[non_trivial],
                                           bins=(bins,bins), range=((0.0, r_max),(0.0, np.pi)) )

    #bin2d, r_edge, t_edge = np.histogram2d(r, theta,
    #                                       bins=(bins,bins), range=((0.0, r_max),(0.0, np.pi)) )
    
    #bin2d = filt.fftConvolve(bin2d, np.ones((3,3)).astype(float)/9.0) #4.0, 4.0, 0.0)
    #bin2d = filt.fftSmooth2d(bin2d, 1.0, 1.0, 0.0)

    wrpos, wtpos = np.where(bin2d > 1)
    avgPerBin = np.mean(bin2d[wrpos,wtpos])
    thresh = max(ncut, navg*avgPerBin)

    locus, numLocus = ndimg.label(bin2d > thresh, structure=np.ones((3,3)))
    print "Threshold: ", len(wrpos), len(theta), avgPerBin, thresh, numLocus

    rs, ts, idx = [], [], []
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
            
        tlo, thi = t_edge[min_ti], t_edge[max_ti+1]
        rlo, rhi = r_edge[min_ri], r_edge[max_ri+1]
        nbox = len(loc_t)
        
        wtmp = np.where((theta >= tlo) & (theta < thi) & (r >= rlo) & (r < rhi))[0]
        t_tmp = np.median(theta[wtmp])
        r_tmp = np.median(r[wtmp])
        print "%6.1f %6.1f  %6.3f %6.3f  %6.1f %6.1f   %6.3f %6.1f  %3d  %3d" % (loc_t.mean(), loc_r.mean(), 0.5*(tlo + thi), thi-tlo, 0.50*(rlo+ rhi), rhi-rlo, t_tmp, r_tmp,  len(wtmp), nbox)
        rs.append(r_tmp)
        ts.append(t_tmp)
        idx.append(wtmp)

    return bin2d, r_edge, t_edge, rs, ts, idx



    
    
if __name__ == '__main__':

    r0s = [200, 400]
    theta0s = [np.pi/3.0, np.pi/7.0]

    dtheta = 0.05

    bins = 200
    n = 200
    nx, ny = 512, 512
    
    # r = x*cos(t) + y*sin(t)
    # --> y = (r - x*cos(t))/sin(t)
    #x0 = nx*np.arange(0, n)/n
    x0 = 0.5*nx + 3.0*np.random.normal(size=n//8)
    x0 = np.append(x0, 0.6*nx + 3.0*np.random.normal(size=n//32))
    xx = np.array([])
    yy = np.array([])
    theta = np.array([])
    for r0, theta0 in zip(r0s, theta0s):
        x = x0.copy()
        y = (r0 - x*np.cos(theta0))/np.sin(theta0)
        w = np.where((y > 0) & (y < ny))[0]
        n = len(w)
        xx = np.append(xx, x[w])
        yy = np.append(yy, y[w])

        # add a random error to theta
        theta = np.append(theta, theta0 + dtheta*np.random.normal(size=n))
    r_max = max(xx.max(), yy.max())
        
    print "N: ", n

    # add some totally fake garbage
    xn = np.random.uniform(nx, size=n)
    yn = np.random.uniform(ny, size=n)
    tn = np.random.uniform(np.pi, size=n)

    xx = np.append(xx, xn)
    yy = np.append(yy, yn)
    theta = np.append(theta, tn)
    
    r_new, t_new, r, _xx, _yy = hesse_iter(theta, xx, yy, niter=8)
    print t_new.mean(), r_new.mean()
    bin2d, r_edge, t_edge, rs, ts, idx = hesse_bin(r_new, t_new, bins=bins, r_max=r_max)

    print rs, ts, bin2d.shape
    bin2d0, r_edge, t_edge = np.histogram2d(r, theta, bins=(bins,bins), range=((0.0, r_max),(0.0, np.pi)) )
    

    
    fig = figure.Figure()
    canvas = FigCanvas(fig)   
    ax = fig.add_subplot(321)
    xlo, xhi, ylo, yhi = 0, 512, 0, 512
    ax.plot(xx, yy, 'r.')
    ax.set_xlim([xlo, xhi])
    ax.set_ylim([ylo, yhi])

    ax = fig.add_subplot(322)
    ax.imshow(bin2d0, cmap='gray_r', origin='bottom', extent=(0.0, np.pi, 0.0, r_max), aspect='auto', interpolation='none')
    ax.scatter(theta, r, c='r', s=20.0, marker='.', edgecolor='none')
    ax.scatter(t_new, r_new, c='g', s=20.0, marker='.', edgecolor='none')
    ax.set_xlim([min(theta0s)-0.2, max(theta0s)+0.2])
    ax.set_ylim([min(r0s) - 40, max(r0s) + 40])

    i = 0
    for r0, theta0 in zip(r0s, theta0s):
        ax = fig.add_subplot(3,2,3 + i)
        i += 1
        ax.imshow(bin2d, cmap='gray_r', origin='bottom', extent=(0.0, np.pi, 0.0, r_max), aspect='auto', interpolation='none')
        colors = 'r', 'b', 'c'
        for i in range(len(ts)):
            ax.scatter(theta[idx[i]], r[idx[i]], c=colors[i], s=20.0, marker='.', edgecolor='none')
            ax.scatter(ts, rs, c='g', s=30.0, marker='.', edgecolor='none')
        ax.set_xlim([theta0-0.1, theta0+0.1])
        ax.set_ylim([r0 - 20, r0 + 20])

    fig.savefig("hesse.png")   

    #fig = figure.Figure()
    #canvas = FigCanvas(fig)
    #ax = fig.add_subplot(111)
    #ax.imshow(bin2d, cmap='gray_r', origin='bottom', interpolation='none', extent=(0.0, np.pi, 0.0, r_max), aspect='auto')

    #fig.savefig("test.png")   
    
