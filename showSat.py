#!/usr/bin/env python

import sys, os, re
import argparse

import numpy as np
import matplotlib.figure as figure
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
import lssttools.utils as lsstutil

def main(rerun, visit):

    bin = 16
    
    # load the known sat file
    filename = "testvisits.dat"
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    ccds = []
    for line in lines:
        m = re.match("^"+str(visit), line)
        if m:
            fields = line.split()
            v, f, n = fields[0:3]
            if n > 0:
                types = fields[3].split(',')
                for i, t in enumerate(types):
                    ccds += [int(re.sub("s", "", c)) for c in fields[4+i].split(',')]
                    
    present  = set(ccds)
    absent   = set(range(104)) - present
    
    logs = [(c, "data/%d/log%05d-%03d.txt" % (visit, visit ,c)) for c in range(104)]
    positive = set()
    for c, log  in logs:
        with open(log, 'r') as lp:
            for line in lp.readlines():
                m = re.search("Detected ([123456789]) satellite", line)
                if m:
                    positive.add(c)
    negative = set(range(104)) - positive
    
    # make plot
    butler = lsstutil.getButler(rerun, root=None)
    fimg = lsstutil.FpaImage(butler.get('camera'), scale=bin)

    tp = present & positive, 1.8
    tn = absent  & negative, 2.4
    fn = present & negative, 4.0
    fp = absent  & positive, 0.1

    recall = 1.0*len(tp[0]) / (len(tp[0]) + len(fn[0]))
    precision = 1.0*len(tp[0]) / (len(tp[0]) + len(fp[0]))

    f1 = 2.0 * (recall*precision)/(recall + precision)
    print "Precision/Recall/f1: ", precision, recall, f1
    
    fimg.image -= 0.1
    for s,v in (tp, tn, fn, fp):
        for c in s:
            #print c, v
            img = fimg.getPixels(c)
            img += v

    fig = figure.Figure()
    canvas = FigCanvas(fig)
    ax = fig.add_subplot(111)

    cmap = cm.gist_rainbow
    cmap.set_under('w', 0.0)
    ax.imshow(fimg.image[::-1], cmap=cmap, vmin=0.0, vmax=4.0)

    png = "fpa-%05d.png" % (visit)
    fig.savefig(png)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rerun", help="rerun")
    parser.add_argument("visit", type=int, help="Visit number")
    args = parser.parse_args()
    main(args.rerun, args.visit)
