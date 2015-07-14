#!/usr/bin/env python

import argparse
import collections
import cPickle as pickle
import numpy as np
import lsst.daf.persistence as dafPersist
import hsc.pipe.base.butler as hscButler
import satelliteTrail as satTrail
import SatelliteTask as satTask

import mapreduce as mapr

colors = {
    "red"    :"31",
    "green"  :"32",
    "yellow" :"33",
    "blue"   :"34",
    "magenta":"35",
    "cyan"   :"36",
    "grey"   :"37",
    }


def color(text, color, bold=False):
    base = "\033["
    code = colors[color]
    if bold:
        code += ";1"
    prefix = base + code + "m"
    suffix = base + "0m"
    return prefix + text + suffix


class Candidate(object):
    
    positiveKinds = set(("satellite", "aircraft", "vertical", "meteor"))
    negativeKinds = set(("empty","scattered"))
    validKinds    = positiveKinds | negativeKinds
    
    def __init__(self, kind, visit, ccd, trail):
        if kind not in self.validKinds:
            raise ValueError("Candidate 'kind' must be: ", self.validKinds)
            
        self.kind  = kind
        self.visit = visit
        self.ccd   = ccd
        self.trail = trail

        
knownCandidates = [
    
    # satellites
    Candidate("satellite",  242, 95, satTrail.SatelliteTrail(r=1497.8, theta=1.245, width=21.12)),
    Candidate("satellite",  258, 98, satTrail.SatelliteTrail(r=197.5,  theta=6.199, width=23.45)),
    Candidate("satellite",  258, 99, satTrail.SatelliteTrail(r=574.0,  theta=6.219, width=18.70)),
    Candidate("satellite",  260, 96, satTrail.SatelliteTrail(r=42.4,   theta=3.021, width=16.26)),
    
    Candidate("satellite",  268, 65, satTrail.SatelliteTrail(r=596.2,  theta=2.735, width=13.16)),
    Candidate("satellite",  270, 78, satTrail.SatelliteTrail(r=1195.6, theta=5.871, width=14.58)),
    Candidate("satellite",  272, 73, satTrail.SatelliteTrail(r=586.5,  theta=0.279, width=16.39)),
    Candidate("satellite", 1166, 96, satTrail.SatelliteTrail(r=1177.4, theta=6.162, width=11.57)),
    Candidate("satellite", 1168, 47, satTrail.SatelliteTrail(r=2492.8, theta=1.430, width=12.91)),
    Candidate("satellite", 1184, 78, satTrail.SatelliteTrail(r=2218.0, theta=0.841, width=19.96)),
    Candidate("satellite", 1236, 65, satTrail.SatelliteTrail(r=1580.0, theta=0.286, width= 7.94)),

    # near vertical
    Candidate("vertical",   246, 50, satTrail.SatelliteTrail(r=1553.8, theta=6.272, width=19.10)),
    
    # meteors
    Candidate("meteor",    1184, 84, satTrail.SatelliteTrail(r=3621.9, theta=0.859, width=11.90)),
    
    # aircraft
    Candidate("aircraft",  1166, 65, satTrail.SatelliteTrail(r= 791.9, theta=6.058, width=22.55)),
    Candidate("aircraft",  1166, 70, satTrail.SatelliteTrail(r= 244.4, theta=2.905, width=25.09)),
    Candidate("aircraft",  1188, 18, satTrail.SatelliteTrail(r=1140.0, theta=6.183, width=18.73)),
    Candidate("aircraft",  1240, 51, satTrail.SatelliteTrail(r= 885.4, theta=2.379, width=19.34)),
    Candidate("aircraft",  1248, 43, satTrail.SatelliteTrail(r= 606.7, theta=6.013, width=15.81)),


    # empty ... take one from each visit ... pick CCDs near the center
    Candidate("empty", 242,  18, None),
    Candidate("empty", 244,  19, None),
    Candidate("empty", 246,  25, None),
    Candidate("empty", 248,  26, None),
    Candidate("empty", 250,  33, None),
    Candidate("empty", 254,  34, None),
    Candidate("empty", 256,  41, None),
    Candidate("empty", 258,  42, None),
    Candidate("empty", 260,  49, None),
    Candidate("empty", 262,  50, None),
    Candidate("empty", 264,  57, None),
    Candidate("empty", 266,  58, None),
    Candidate("empty", 268,  64, None),
    Candidate("empty", 270,  66, None),   
    Candidate("empty", 272,  74, None),
    Candidate("empty", 1166,  74, None),
    Candidate("empty", 1168,  80, None),
    Candidate("empty", 1170,  81, None),
    Candidate("empty", 1172,  18, None),
    Candidate("empty", 1174,  19, None),
    Candidate("empty", 1176,  25, None),
    Candidate("empty", 1178,  26, None),
    Candidate("empty", 1180,  33, None),
    Candidate("empty", 1182,  34, None),
    Candidate("empty", 1184,  41, None),
    Candidate("empty", 1186,  42, None),
    Candidate("empty", 1188,  49, None),
    Candidate("empty", 1190,  50, None),
    Candidate("empty", 1192,  57, None),
    Candidate("empty", 1194,  58, None),
    Candidate("empty", 1202,  65, None),
    Candidate("empty", 1204,  66, None),
    Candidate("empty", 1206,  73, None),
    Candidate("empty", 1208,  74, None),        
    Candidate("empty", 1210,  80, None),
    Candidate("empty", 1212,  81, None),
    Candidate("empty", 1214,  18, None),
    Candidate("empty", 1216,  19, None),
    Candidate("empty", 1218,  25, None),
    Candidate("empty", 1220,  26, None),
    Candidate("empty", 1222,  33, None),
    Candidate("empty", 1228,  34, None),
    Candidate("empty", 1230,  41, None),
    Candidate("empty", 1232,  42, None),
    Candidate("empty", 1236,  49, None),
    Candidate("empty", 1238,  50, None),
    Candidate("empty", 1240,  57, None),
    Candidate("empty", 1242,  58, None),
    Candidate("empty", 1244,  65, None),
    Candidate("empty", 1246,  66, None),
    Candidate("empty", 1248,  73, None),
    Candidate("empty", 1886,  74, None),
    Candidate("empty", 1888,  80, None),
    Candidate("empty", 1890,  81, None),


    # previous false positives (scattered light)
    Candidate("scattered",  248, 15, None),
    Candidate("scattered",  260, 98, None),
    
]

if False:
    knownCandidates = [
        Candidate("aircraft",  1166, 65, satTrail.SatelliteTrail(r= 791.9, theta=6.058, width=22.55)),
        Candidate("satellite", 1236, 65, satTrail.SatelliteTrail(r=1580.0, theta=0.286, width= 7.94)),    
        Candidate("empty",     1236, 50, None),
    ]
    
if False:
    knownCandidates = [
        Candidate("empty", 1236, 50, None),
    ]



def hashDataId(dataId):
    return "%04d-%03d" % (int(dataId['visit']), int(dataId['ccd']))
    
def process(dataRef):
    task                  = satTask.SatelliteTask()
    foundTrails, runtime  = task.run(dataRef)
    return (hashDataId(dataRef.dataId), foundTrails, runtime)

    
Event = collections.namedtuple("Event", "input detected")
class EventList(list):

    @property
    def positiveDetections(self):
        return 1.0*sum(1 for input,detected in self if detected)
    @property
    def negativeDetections(self):
        return 1.0*sum(1 for input,detected in self if not detected)
    @property
    def positiveInputs(self):
        return 1.0*sum(1 for input,detected in self if input)
    @property
    def negativeInputs(self):
        return 1.0*sum(1 for input,detected in self if not input)
        
    @property
    def truePositives(self):
        return 1.0*sum(1 for input,detected in self if input and detected)
    @property
    def trueNegatives(self):
        return 1.0*sum(1 for input,detected in self if (not input) and (not detected))
    @property
    def falsePositives(self):
        return 1.0*sum(1 for input,detected in self if (not input) and detected)
    @property
    def falseNegatives(self):
        return 1.0*sum(1 for input,detected in self if input and (not detected))

    @property
    def recall(self):
        posIn = self.positiveInputs
        if posIn == 0:
            #print "%s: No positive inputs.  Recall is undefined." % (color("WARNING", "yellow"))
            posIn = -1
        return self.truePositives/posIn
    @property
    def precision(self):
        posDet = self.positiveDetections
        if posDet == 0:
            #print "%s: No positive detections.  Precision is undefined." % (color("WARNING", "yellow"))
            posDet = -1
        return self.truePositives/posDet
    @property
    def f1(self):
        denom = self.recall + self.precision
        if len(self) == 0 or denom == 0.0:
            return 0.0
        return 2.0*(self.recall*self.precision)/denom
        



        
def main(root, threads, output, input=None, kind=None):

    butler       = dafPersist.Butler(root)
    rMax, thetaMax = 20.0, 0.15
        
    allMessages = ""

    ###################################################################
    # Create an eventList for all kinds of candiate Trail
    eventLists = {}
    nMasked = {}
    for k in Candidate.validKinds:
        eventLists[k] = EventList()
        nMasked[k] = 0
    eventLists['all'] = EventList()
    nMasked['all'] = 0

    candidateLookup = {}
    for candidate in knownCandidates:
        did = {'visit': candidate.visit, 'ccd':candidate.ccd }
        candidateLookup[hashDataId(did)] = candidate

    
    ###################################################################
    # Add jobs to the map    
    if input is None:
        mp = mapr.MapFunc(threads, process)
        for candidate in knownCandidates:
            if kind is None or candidate.kind == kind:
                dataId       = {'visit': candidate.visit, 'ccd': candidate.ccd}
                dataRef      = hscButler.getDataRef(butler, dataId)
                mp.add(dataRef)
        results = mp.run()

    else:
        with open(input, 'r') as fp:
            results = pickle.load(fp)

    if output is not None:
        with open(output, 'w') as fp:
            pickle.dump(results, fp)
    
    ####################################################################
    # Tally the results and see how we did
    runtimes = []
    for result in results:
        dataHash, foundTrails, runtime = result
        runtimes.append(runtime)
        candidate = candidateLookup[dataHash]
        
        nTrail = len(foundTrails)

        eList = EventList()
        
        t = candidate.trail
        result = ""
        claimed = [False]*nTrail
        for iTrail,fTrail in enumerate(foundTrails):
            
            nMasked[candidate.kind] += fTrail.nMaskedPixels
            nMasked['all'] += fTrail.nMaskedPixels
                    
            if t  and fTrail.isNear(t, rMax, thetaMax):
                result += "\n  %s: %s" % (color("TRUE-POS", "green"), fTrail)
                claimed[iTrail] = True
                eList.append(Event(True, True))
                
        if not result:
            if t is None:
                result += "\n  %s: %s" % (color("TRUE-NEG", "green"), "No trails present, and none found.")
                eList.append(Event(False, False))
            else:
                result += "\n  %s: %s" % (color("FALSE-NEG", "red"), t)
                eList.append(Event(True, False))
            
        nUnclaimed = len(claimed) - sum(claimed)
        if nUnclaimed > 0:
            result += "\n  %s: %d Unclaimed trails (total=%d).\n" % \
                      (color("FALSE-POS", "red"), nUnclaimed, nTrail)
            for iClaim,claim in enumerate(claimed):
                if not claim:
                    result  += "  --> Unclaimed: %s\n" % (color(str(foundTrails[iClaim]), 'yellow'))
                    eList.append(Event(False, True))
                    
        did = "(%d, %d)" % (candidate.visit, candidate.ccd)
        if t:
            msg = "%s [r=%.1f theta=%.3f wid=%.1f]." % (color(did, "cyan"), t.r, t.theta, t.width)
        else:
            msg = "%s [empty candidate]" % (color(did, "cyan"))
        msg += result
        allMessages += msg+"\n"
        print "\n"+msg+"\n"
        eventLists[candidate.kind] += eList
        eventLists['all'] += eList
        
    print color("=== Summary ===", "magenta")
    print allMessages

    for kind, eventList in eventLists.items():
        print color("=== %s ===" % (kind), "magenta")
        if len(eventList) == 0:
            print "No events."
            continue
        nPos = eventList.positiveDetections or 1
        print "Recall,precision,f1:  %4.2f %4.2f  %4.2f"%(eventList.recall, eventList.precision, eventList.f1)
        print "Masked pixels: %10d" % (nMasked[kind]), \
            " (all: n = %d, %.2f%%)" % (len(eventList),100.0*nMasked[kind]/(len(eventList)*2048*4096)), \
            " (det: n = %d, %.2f%%)" % (eventList.positiveDetections,100.0*nMasked[kind]/(nPos*2048*4096))
        print "\n"

    rt = np.array(runtimes)
    print "Runtimes:   mean=%.2f  med=%.2f  std=%.2f  min=%.2f  max=%.2f\n" % \
        (rt.mean(), np.median(rt), rt.std(), rt.min(), rt.max())
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("-j", "--threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("-k", "--kind", choices=Candidate.validKinds, help="Specify kind to run.")
    parser.add_argument("-o", "--output", default="known.pickle")
    parser.add_argument("-i", "--input", default=None)
    args = parser.parse_args()
    
    if args.output == 'None':
        args.output = None
        
    main(args.root, args.threads, args.output, args.input, kind=args.kind)
