#!/usr/bin/env python

import sys, os, re
import argparse
import collections
import datetime
import numpy as np
import matplotlib.figure as figure
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
import lssttools.utils as lsstutil

def main(rerun, visits):

    bin = 16
    
    # load the known sat file
    filename = "testvisits.dat"
    with open(filename, 'r') as fp:
        lines = fp.readlines()

    ccds = collections.defaultdict(list)
    
    for line in lines:
        line = re.sub('#.*', '', line).strip()
        if len(line) == 0:
            continue
        
        fields = line.split()

        v, f, n = fields[0:3]
        if int(n) > 0:
            types = fields[3].split(',')
            for i, t in enumerate(types):
                ccds[int(v)] += [int(re.sub("s", "", c)) for c in fields[4+i].split(',')]
        
    tps, tns, fps, fns = 0, 0, 0, 0
    for visit in visits:

        present  = set(ccds[visit])
        absent   = set(range(104)) - present

        logs = [(c, "data/%d/log%05d-%03d.txt" % (visit, visit ,c)) for c in range(104)]
        positive = set()
        for c, log  in logs:
            if os.path.exists(log):
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

        tps += len(tp[0])
        tns += len(tn[0])
        fps += len(fp[0])
        fns += len(fn[0])

        all_predicted = len(positive)
        all_occurred  = len(present)

        if all_occurred:
            recall    = 1.0*len(tp[0]) / all_occurred
        else:
            recall    = 1.0
        if all_predicted:
            precision = 1.0*len(tp[0]) / all_predicted
        else:
            precision = 1.0

        if recall + precision:
            f1 = 2.0 * (recall*precision)/(recall + precision)
        else:
            f1 = 0.0
            
        print "%d  Precision/Recall/f1: ", visit, precision, recall, f1

        fimg.image -= 0.1
        for s,v in (tp, tn, fn, fp):
            for c in s:
                #print c, v
                img = fimg.getPixels(c)
                img += v

        fig = figure.Figure()
        canvas = FigCanvas(fig)
        ax = fig.add_subplot(111)

        ax.set_title( str(visit) + "    "+ datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S") )
        cmap = cm.gist_rainbow
        cmap.set_under('w', 0.0)
        ax.imshow(fimg.image[::-1], cmap=cmap, vmin=0.0, vmax=4.0)

        png = "fpa-%05d.png" % (visit)
        fig.savefig(png)


        
    recall = 1.0*tps/ (tps + fns)
    precision = 1.0*tps / (tps + fps)

    f1 = 2.0 * (recall*precision)/(recall + precision)

    print "Final Precision/Recall/f1:", precision, recall, f1

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rerun", help="rerun")
    parser.add_argument("visit", help="Visit number")
    args = parser.parse_args()
    main(args.rerun, [int(x) for x in lsstutil.idSplit(args.visit)])
