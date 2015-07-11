#!/usr/bin/env python

import argparse
import collections
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
    
    positiveKinds = set(("satellite", "aircraft"))
    negativeKinds = set(("empty",))
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
    Candidate("satellite", 1236, 65, satTrail.SatelliteTrail(r=1580.0, theta=0.286, width= 7.94)),
    Candidate("satellite",  242, 95, satTrail.SatelliteTrail(r=1497.8, theta=1.245, width=21.12)),
    Candidate("satellite",  270, 78, satTrail.SatelliteTrail(r=1195.6, theta=5.871, width=14.58)),
    Candidate("satellite", 1184, 78, satTrail.SatelliteTrail(r=2218.0, theta=0.841, width=19.96)),
    Candidate("satellite", 1168, 47, satTrail.SatelliteTrail(r=2492.8, theta=1.430, width=12.91)),
    Candidate("satellite", 1166, 96, satTrail.SatelliteTrail(r=1177.4, theta=6.162, width=11.57)),

    # meteors
    
    # aircraft
    Candidate("aircraft", 1166, 65, satTrail.SatelliteTrail(r= 791.9, theta=6.058, width=22.55)),
    Candidate("aircraft", 1166, 70, satTrail.SatelliteTrail(r= 244.4, theta=2.905, width=25.09)),
    Candidate("aircraft", 1188, 18, satTrail.SatelliteTrail(r=1140.0, theta=6.183, width=18.73)),
    Candidate("aircraft", 1240, 51, satTrail.SatelliteTrail(r= 885.4, theta=2.379, width=19.34)),
    Candidate("aircraft", 1248, 43, satTrail.SatelliteTrail(r= 606.7, theta=6.013, width=15.81)),

    # empty
    Candidate("empty",    1236, 50, None),


    # previous false positives (scattered light)
    Candidate("scattered",) 
    
]

if False:
    knownCandidates = [
        Candidate("satellite", 1236, 65, satTrail.SatelliteTrail(r=1580.0, theta=0.286, width= 7.94)),    
        Candidate("aircraft",  1166, 65, satTrail.SatelliteTrail(r= 791.9, theta=6.058, width=22.55)),
        Candidate("empty",     1236, 50, None),
    ]
    
if False:
    knownCandidates = [
        Candidate("empty", 1236, 50, None),
    ]


def process(dataRef, candidate):
    task         = satTask.SatelliteTask()
    foundTrails  = task.run(dataRef)
    return (candidate, foundTrails)



    
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
            print "%s: No positive inputs.  Recall is undefined." % (color("WARNING", "yellow"))
            posIn = -1
        return self.truePositives/posIn
    @property
    def precision(self):
        posDet = self.positiveDetections
        if posDet == 0:
            print "%s: No positive detections.  Precision is undefined." % (color("WARNING", "yellow"))
            posDet = -1
        return self.truePositives/posDet
    @property
    def f1(self):
        return 2.0*(self.recall*self.precision)/(self.recall + self.precision)
        



        
def main(root, threads):

    butler       = dafPersist.Butler(root)
    rMax, thetaMax = 20.0, 0.15
        
    allMessages = ""

    mp = mapr.MapFunc(threads, process)

    events = []
    for candidate in knownCandidates:
        dataId       = {'visit': candidate.visit, 'ccd': candidate.ccd}
        dataRef      = hscButler.getDataRef(butler, dataId)
        mp.add(dataRef, candidate)
    results = mp.run()

    eventList = EventList()
    
    for result in results:
        candidate, foundTrails = result
        nTrail = len(foundTrails)

        eList = EventList()
        
        t = candidate.trail
        result = ""
        claimed = [False]*nTrail
        for iTrail,fTrail in enumerate(foundTrails):
            if fTrail.isNear(t, rMax, thetaMax):
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
        eventList += eList
        
    print color("=== Summary ===", "magenta")
    print allMessages
    print "Recall,precision,f1:", eventList.recall, eventList.precision, eventList.f1
    


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("-j", "--threads", type=int, default=1, help="Number of threads to use")
    args = parser.parse_args()
    main(args.root, args.threads)
