#!/usr/bin/env python

import sys, os, re
import copy
import StringIO as sio
import argparse
import collections
import datetime
import numpy as np
import matplotlib.figure as figure
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
import lssttools.utils as lsstutil

class Trail(object):
    def __init__(self, r, theta, candpix, binmax, npix):
        self.r = float(r)
        self.theta = float(theta)
        self.candpix = int(candpix)
        self.binmax = int(binmax)
        self.npix = int(npix)

class Detection(object):
    def __init__(self, binning, candpix, binmax):
        self.binning = int(binning)
        self.candpix = int(candpix)
        self.binmax = int(binmax)
        self.trails = []
        
class Stat(object):
    def __init__(self, v, c):
        self.visit = int(v)
        self.ccd = int(c)
        self.knownTrails    = 0
        self.detectedTrails = 0
        self.detections = []
        
    def __str__(self):
        s = "%05d %03d %d %d " % (self.visit, self.ccd, self.knownTrails, self.detectedTrails)
        for d in self.detections:
            npix = 0
            for t in d.trails:
                npix += t.npix
            s += " %d   %5d %7d " % (d.binning, d.binmax, npix)
        if len(self.detections) == 1:
            s += " %d   %5d %7d " % (4, 0, 0)
        if len(self.detections) == 0:
            for b in 2,4:
                s += " %d   %5d %7d " % (b, 0, 0)
        return s

def vc(v, c):
    return "%05d-%03d" % (int(v), int(c))

    
def toArray(statList):

    s = ""
    for k,stat in sorted(statList.items()):
        q = str(stat)
        s += q + "\n"

    strio = sio.StringIO(s)
    data = np.genfromtxt(strio, dtype=None)

    return data
    
        
def main(rerun, visits):

    bin = 32
    
    # load the known sat file
    filename = "testvisits.dat"
    with open(filename, 'r') as fp:
        lines = fp.readlines()

    all_types = "bfpvaF"
    ccds = {}
    tps, fps, tns, fns = {}, {}, {}, {}
    for c in all_types:
        ccds[c] = collections.defaultdict(list)
        tps[c], fps[c], tns[c], fns[c] = 0, 0, 0, 0


    statList = {}

    for line in lines:
        line = re.sub('#.*', '', line).strip()
        if len(line) == 0:
            continue
        
        fields = line.split()

        v, f, n = fields[0:3]
        if int(v) not in visits:
            continue
        for c in range(104):
            statList[vc(v,c)] = Stat(v, c)
        if int(n) > 0:
            types = fields[3].split(',')
            for i, t in enumerate(types):
                cs = [int(re.sub("[cs]", "", c)) for c in fields[4+i].split(',')]                
                ccds[t][int(v)] += cs
                for c in cs:
                    statList[vc(v,c)].knownTrails += 1

    stats = []
                
    for visit in visits:

        print "Running", visit
        
        logs = [(c, "data/%04d/log%05d-%03d.txt" % (visit, visit ,c)) for c in range(104)]
        all_positive = set()
        for c, log  in logs:
            if os.path.exists(log):
                with open(log, 'r') as lp:
                    for line in lp.readlines():
                        m = re.search("\(binned (\d+)x\)", line)
                        if m:
                            binning = m.groups()[0]
                        m = re.search("Detected (\d+) satellite trails.  cand-pix: (\d+) bin-max: (\d+)", line)
                        if m:
                            nsat, cpix, bm = m.groups()
                            if int(nsat) > 0:
                                all_positive.add(c)
                            stat = statList[vc(visit,c)]
                            stat.detectedTrails += int(nsat)
                            det = Detection(binning, cpix, bm)
                            stat.detections.append(det)
                        m = re.search("Trail (\d+) of (\d+) \(r: (\d+\.\d),theta: (\d\.\d+)\):  cand-pix: (\d+) max-bin-count: (\d+) mask-pix: (\d+)", line)
                        if m:
                            n1, n2, r, theta, cpix, binmax, npix = m.groups()
                            trail = Trail(r, theta, cpix, binmax, npix)
                            det.trails.append(trail)
                            
        all_negative = set(range(104)) - all_positive

        
        for i_t, track_type in enumerate(all_types):
        
            present  = set(ccds[track_type][visit])
            absent   = set(range(104)) - present

            # only consider positives if they're not expected in one of the other types
            # for detection, we don't know what 'type' we got.
            positive = copy.copy(all_positive)
            for i_tt, track_tmp in enumerate(all_types):
                if i_t == i_tt:
                    continue
                # in this type, but not in our type
                s = set(ccds[track_tmp][visit]) - set(ccds[track_type][visit])                
                positive -= s
            negative = set(range(104)) - positive
            
            tp = present & positive, 1.8
            tn = absent  & negative, 2.4
            fn = present & negative, 4.0
            fp = absent  & positive, 0.1

            tps[track_type] += len(tp[0])
            tns[track_type] += len(tn[0])
            fps[track_type] += len(fp[0])
            fns[track_type] += len(fn[0])


    data = toArray(statList)
    wTP, = np.where( (data[:,2] > 0)  & (data[:,3] > 0) )
    wFP, = np.where( (data[:,2] == 0) & (data[:,3] > 0) )
    wTN, = np.where( (data[:,2] == 0) & (data[:,3] == 0) )
    wFN, = np.where( (data[:,2] > 0)  & (data[:,3] == 0) )

    print len(wTP), len(wFP)

    fig = figure.Figure()
    can = FigCanvas(fig)
    ax2 = fig.add_subplot(211)
    ax4 = fig.add_subplot(212)
    nbins = 20
    labels = "tp", "fp", "tn", "fn"
    colors = "g", "r", "c", "m"
    for i,w in enumerate((wTP, wFP, wTN, wFN)):
        if len(w) > 0:
            d2 = data[w,5]
            d4 = data[w,8]
        else:
            d2 = [0]
            d4 = [0]
        bins = 10**((np.log10(1000)/20) * np.arange(20))
        ax2.hist(d2, alpha=0.5, bins=bins, label=labels[i], color=colors[i])
        ax4.hist(d4, alpha=0.5, bins=bins, label=labels[i], color=colors[i])
    ax2.set_xscale('log')
    #ax2.set_yscale('log')
    ax2.set_xlim([1, 1000])
    ax2.set_ylim([0, 100])
    ax4.set_xscale('log')
    #ax4.set_yscale('log')
    ax4.set_xlim([1, 1000])
    ax4.set_ylim([0, 100])
    ax2.legend()

    fig.savefig("hist.png")

    
    for track_type in all_types:
        
        n_positive = tps[track_type] + fps[track_type] + 0.001
        n_trail    = tps[track_type] + fns[track_type] + 0.001
        recall    = 1.0*tps[track_type]/ n_trail
        precision = 1.0*tps[track_type] / n_positive

        f1 = 2.0 * (recall*precision)/(recall + precision + 0.001)

        print "Final (type=%s) Precision/Recall/f1:" % (track_type), precision, recall, f1


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rerun", help="rerun")
    parser.add_argument("visit", help="Visit number")
    args = parser.parse_args()
    main(args.rerun, [int(x) for x in lsstutil.idSplit(args.visit)])
