#!/usr/bin/env python
import satelliteTrail as satTrail

class Candidate(object):
    
    positiveKinds = set(("satellite", "aircraft", "meteor", "diffraction"))
    negativeKinds = set(("empty","scattered"))
    ignoredKinds  = set(("moustache",))
    validKinds    = positiveKinds | negativeKinds | ignoredKinds
    
    def __init__(self, kind, visit, ccd, trail):
        if kind not in self.validKinds:
            raise ValueError("Candidate 'kind' must be: ", self.validKinds)
            
        self.kind  = kind
        self.visit = visit
        self.ccd   = ccd
        self.trail = trail

    def __eq__(self, other):
        if (self.trail is None) or (other.trail is None):
            return False
        return (np.abs(self.trail.r - other.trail.r) < 0.01) and \
            (np.abs(self.trail.theta - other.trail.theta) < 0.01)


        
knownCandidates = [

#    Candidate("satellite",  ,  , satTrail.SatelliteTrail()),
    
    # 242 
    Candidate("satellite",  242,  2, satTrail.SatelliteTrail(r=2055.0,theta=1.290,width=23.25)),
    Candidate("moustache",  242,  4, satTrail.SatelliteTrail(r=1910.6,theta=2.360,width=36.08)),
    Candidate("satellite",  242,  7, satTrail.SatelliteTrail(r=2643.3,theta=1.286,width=23.21)),
    Candidate("moustache",  242, 15, satTrail.SatelliteTrail(r=2083.6,theta=0.979,width=46.27)),
    Candidate("satellite",  242, 19, satTrail.SatelliteTrail(r=3849.5,theta=1.278,width=16.21)),
    Candidate("moustache",  242, 21, satTrail.SatelliteTrail(r=3213.8,theta=0.797,width=43.33)),
    Candidate("scattered",  242, 32, satTrail.SatelliteTrail(r=54.2,theta=1.577,width=49.04)),
    Candidate("satellite",  242, 35, satTrail.SatelliteTrail(r=795.2,theta=1.275,width=25.48)),
    Candidate("satellite",  242, 43, satTrail.SatelliteTrail(r=1419.1,theta=1.271,width=31.58)),
    Candidate("satellite",  242, 51, satTrail.SatelliteTrail(r=2051.1,theta=1.265,width=16.92)),
    Candidate("satellite",  242, 59, satTrail.SatelliteTrail(r=2683.1,theta=1.268,width=29.32)),
    Candidate("satellite",  242, 67, satTrail.SatelliteTrail(r=3323.1,theta=1.262,width=20.21)),
    Candidate("satellite",  242, 75, satTrail.SatelliteTrail(r=3974.4,theta=1.258,width=25.26)),
    Candidate("satellite",  242, 83, satTrail.SatelliteTrail(r=370.9,theta=1.254,width=15.31)),
    Candidate("satellite",  242, 89, satTrail.SatelliteTrail(r=821.9,theta=1.252,width=16.19)),    
    Candidate("satellite",  242, 95, satTrail.SatelliteTrail(r=1497.8, theta=1.245, width=21.12)),

    # 246
    Candidate("satellite",  246, 48, satTrail.SatelliteTrail(r=1472.9,theta=6.274,width=29.55)),
    Candidate("satellite",  246, 49, satTrail.SatelliteTrail(r=1512.8,theta=6.272,width=21.52)),
    Candidate("satellite",  246, 50, satTrail.SatelliteTrail(r=1572.7,theta=6.277,width=24.05)),
    Candidate("satellite",  246, 51, satTrail.SatelliteTrail(r=1596.7,theta=6.268,width=22.75)),
    Candidate("satellite",  246, 52, satTrail.SatelliteTrail(r=1650.5,theta=6.271,width=18.88)),
    Candidate("scattered",  246, 95, satTrail.SatelliteTrail(r=1174.9,theta=2.369,width=24.37)),

    # 248
    Candidate("scattered",  248, 15, None),
    Candidate("satellite",  248, 46, satTrail.SatelliteTrail(r=1885.1,theta=6.274,width=28.09)),

    # 258
    Candidate("satellite",  258, 98, satTrail.SatelliteTrail(r=197.5,  theta=6.199, width=23.45)),
    Candidate("satellite",  258, 99, satTrail.SatelliteTrail(r=574.0,  theta=6.219, width=18.70)),

    # 260
    Candidate("satellite",  260, 19, satTrail.SatelliteTrail(r=1900.7,theta=0.325,width=25.64)),
    Candidate("satellite",  260, 20, satTrail.SatelliteTrail(r=491.1,theta=0.301,width=15.42)),
    Candidate("satellite",  260, 27, satTrail.SatelliteTrail(r=2508.0,theta=0.295,width=22.08)),
    Candidate("satellite",  260, 28, satTrail.SatelliteTrail(r=1205.4,theta=0.286,width=14.95)),
    Candidate("satellite",  260, 37, satTrail.SatelliteTrail(r=1988.3,theta=0.275,width=15.49)),
    Candidate("satellite",  260, 96, satTrail.SatelliteTrail(r=42.4,theta=3.021,width=16.26)),
    Candidate("scattered",  260, 98, None),


    # 262
    Candidate("satellite",    262,  4, satTrail.SatelliteTrail(r=507.9,theta=0.333,width=16.49)),
    Candidate("aircraft",     262,  6, satTrail.SatelliteTrail(r=873.5,theta=0.465,width=38.73)),
    Candidate("satellite",    262, 10, satTrail.SatelliteTrail(r=2507.8,theta=0.333,width=14.67)),
    Candidate("satellite",    262, 11, satTrail.SatelliteTrail(r=1050.7,theta=0.324,width=17.89)),
    Candidate("satellite",    262, 15, satTrail.SatelliteTrail(r=548.0,theta=6.123,width=18.11)),
    Candidate("satellite",    262, 17, satTrail.SatelliteTrail(r=2991.2,theta=0.322,width=15.55)),
    Candidate("satellite",    262, 18, satTrail.SatelliteTrail(r=1583.3,theta=0.319,width=18.36)),    
    Candidate("scattered",    262, 34, satTrail.SatelliteTrail(r=3043.7,theta=1.571,width=32.31)),
    Candidate("scattered",    262, 37, satTrail.SatelliteTrail(r=216.2,theta=2.367,width=38.06)),
    Candidate("diffraction",  262, 67, satTrail.SatelliteTrail(r=1439.3,theta=1.573,width=31.35)),
    Candidate("diffraction",  262, 75, satTrail.SatelliteTrail(r=3271.7,theta=1.573,width=31.88)),
    Candidate("diffraction",  262, 79, satTrail.SatelliteTrail(r=3999.6,theta=1.565,width=27.00)),

    # 264
    Candidate("moustache",    264,  4, satTrail.SatelliteTrail(r=1903.6,theta=2.360,width=32.41)),
    Candidate("satellite",    264, 11, satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0)), #not det
    Candidate("satellite",    264, 13, satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0)), #not det
    Candidate("satellite",    264, 14, satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0)), #not det
    Candidate("satellite",    264, 15, satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0)), #not det
    Candidate("satellite",    264, 17, satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0)), #not det
    Candidate("satellite",    264, 18, satTrail.SatelliteTrail(r=1166.5,theta=6.160,width=23.82)),
    Candidate("satellite",    264, 19, satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0)), #not det
    Candidate("diffraction",  264, 56, satTrail.SatelliteTrail(r=1880.9,theta=1.580,width=31.61)),
    Candidate("diffraction",  264, 71, satTrail.SatelliteTrail(r=755.6,theta=1.572,width=31.87)),
    Candidate("diffraction",  264, 76, satTrail.SatelliteTrail(r=799.5,theta=1.579,width=30.69)),
    Candidate("satellite",    264, 90, satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0)), #not det
    Candidate("aircraft",     264, 96, satTrail.SatelliteTrail(r=1995.9,theta=0.369,width=37.42)),

    # 266
    Candidate("satellite",    266, 23, satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0)), #not det
    Candidate("moustache",    266, 89, satTrail.SatelliteTrail(r=2184.4,theta=2.134,width=31.40)),
    

    # 268 y 
    Candidate("diffraction", 268,  38, satTrail.SatelliteTrail(r=1663.3,theta=1.564,width=31.29 )), 
    Candidate("satellite",   268,  44, satTrail.SatelliteTrail(r=1261.0,theta=2.748,width=18.18 )),
    Candidate("satellite",   268,  45, satTrail.SatelliteTrail(r=459.0,theta=5.891,width=13.96  )),
    Candidate("satellite",   268,  49, satTrail.SatelliteTrail(r=1357.3,theta=6.277,width=21.08 )),
    Candidate("satellite",   268,  50, satTrail.SatelliteTrail(r=1391.4,theta=6.278,width=18.76 )),
    Candidate("satellite",   268,  51, satTrail.SatelliteTrail(r=1058.5,theta=2.741,width=12.97 )),  #DOUBLE TRAILS
    Candidate("satellite",   268,  51, satTrail.SatelliteTrail(r=1417.6,theta=6.276,width=14.49 )),  # DOUBLE TRAILS
    Candidate("satellite",   268,  52, satTrail.SatelliteTrail(r=689.7,theta=5.887,width=14.58  )),  #DOUBLE TRAILS
    Candidate("satellite",   268,  52, satTrail.SatelliteTrail(r=1453.1,theta=6.277,width=22.96 )),  # DOUBLE TRAILS
    Candidate("satellite",   268,  53, satTrail.SatelliteTrail(r=1488.8,theta=6.276,width=15.00   )),
    Candidate("diffraction", 268,  56, satTrail.SatelliteTrail(r=3678.8,theta=1.575,width=29.23 )),
    Candidate("satellite",   268,  58, satTrail.SatelliteTrail(r=866.9,theta=2.737,width=15.86  )),
    Candidate("satellite",   268,  59, satTrail.SatelliteTrail(r=893.9,theta=5.882,width=13.43  )),
    Candidate("satellite",   268,  65, satTrail.SatelliteTrail(r=596.2,theta=2.735,width=13.16  )),
    Candidate("satellite",   268,  66, satTrail.SatelliteTrail(r=1085.7,theta=5.879,width=17.87 )),
    Candidate("moustache",   268,  69, satTrail.SatelliteTrail(r=965.6,theta=1.749,width=32.88  )),
    Candidate("diffraction", 268,  71, satTrail.SatelliteTrail(r=2629.6,theta=1.571,width=30.24 )),
    Candidate("diffraction", 268,  76, satTrail.SatelliteTrail(r=2648.9,theta=1.578,width=28.00 )),
    Candidate("moustache",   268,  96, satTrail.SatelliteTrail(r=1535.3,theta=0.646,width=31.47 )),
    Candidate("moustache",   268,  98, satTrail.SatelliteTrail(r=929.3,theta=6.050,width=38.06  )),
    Candidate("scattered",   268, 103, satTrail.SatelliteTrail(r=59.7,theta=1.572,width=35.98  )),
    

    # 270
    Candidate("diffraction", 270, 16, satTrail.SatelliteTrail(r=2900.5,theta=1.573,width=30.45)),
    Candidate("diffraction", 270, 19, satTrail.SatelliteTrail(r=1020.0,theta=1.576,width=38.60)),
    Candidate("moustache",   270, 21,  satTrail.SatelliteTrail(r=4308.9,theta=1.109,width=31.14)),
    Candidate("scattered",   270, 27, satTrail.SatelliteTrail(r=1479.9,theta=1.576,width=41.64)),
    Candidate("scattered",   270, 44, satTrail.SatelliteTrail(r=1830.2,theta=2.360,width=35.34)),
    Candidate("satellite",   270, 46, satTrail.SatelliteTrail(r=1765.2,theta=6.280,width=14.06)),
    Candidate("scattered",   270, 62, satTrail.SatelliteTrail(r=1767.4,theta=1.571,width=38.69)),
    Candidate("scattered",   270, 64, satTrail.SatelliteTrail(r=774.1,theta=1.571 ,width=29.52)), 
    Candidate("scattered",   270, 67, satTrail.SatelliteTrail(r=2508.7,theta=1.572,width=31.42)), 
    Candidate("satellite",   270, 71, satTrail.SatelliteTrail(r=0.0,theta=0.0, width=0.0)),             # bright but very short.
    Candidate("satellite",   270, 78, satTrail.SatelliteTrail(r=1195.6,theta=5.871,width=14.58)),
    Candidate("scattered",   270, 82, satTrail.SatelliteTrail(r=551.9,theta=2.360 ,width=32.24)), 
    Candidate("satellite",   270,102, satTrail.SatelliteTrail(r=2234.0,theta=5.869,width=14.07)), 


    # 272
    Candidate("satellite",   272, 59, satTrail.SatelliteTrail(r=1769.0,theta=0.017,width=29.88)),
    Candidate("satellite",   272, 60, satTrail.SatelliteTrail(r=1702.3,theta=6.279,width=14.66)),
    Candidate("satellite",   272, 61, satTrail.SatelliteTrail(r=1702.3,theta=6.279,width=14.0)),   # not detected ... why?
    Candidate("diffraction", 272, 67, satTrail.SatelliteTrail(r=1612.4,theta=1.573,width=30.89)),
    Candidate("diffraction", 272, 70, satTrail.SatelliteTrail(r=3305.4,theta=1.576,width=30.97)),
    Candidate("satellite",   272, 72, satTrail.SatelliteTrail(r=1808.8,theta=0.276,width=16.98)),
    Candidate("satellite",   272, 73, satTrail.SatelliteTrail(r=586.5,theta=0.279,width=16.39 )),
    Candidate("diffraction", 272, 75, satTrail.SatelliteTrail(r=3445.2,theta=1.577,width=28.99)),
    Candidate("satellite",   272, 80, satTrail.SatelliteTrail(r=2626.5,theta=0.283,width=15.34)),
    Candidate("satellite",   272, 81, satTrail.SatelliteTrail(r=1436.5,theta=0.290,width=15.36)),
    Candidate("satellite",   272, 82, satTrail.SatelliteTrail(r=20.,theta=0.290,width=15.)),        #tiny ... in the corner
    Candidate("satellite",   272, 88, satTrail.SatelliteTrail(r=2125.4,theta=0.302,width=30.69)),
    Candidate("satellite",   272, 89, satTrail.SatelliteTrail(r=792.1,theta=0.310,width=18.53 )),
    Candidate("satellite",   272, 95, satTrail.SatelliteTrail(r=0.0,theta=0.0,width=0.0)),          # short stub ... never detected
    

    # 1166
    Candidate("satellite", 1166,10   ,satTrail.SatelliteTrail(r=2162.0,theta=0.304,width=15.94 )),
    Candidate("satellite", 1166,11   ,satTrail.SatelliteTrail(r=824.8,theta=0.295, width=11.28 )),
    Candidate("satellite", 1166,17   ,satTrail.SatelliteTrail(r=2788.1,theta=0.294,width=14.48 )),
    Candidate("satellite", 1166,18   ,satTrail.SatelliteTrail(r=1490.3,theta=0.279,width=21.65 )),
    Candidate("moustache", 1166,21   ,satTrail.SatelliteTrail(r=2960.4,theta=0.798,width=23.31 )),
    Candidate("aircraft",  1166,52   ,satTrail.SatelliteTrail(r=398.3,theta=2.935,width=23.83)),
    Candidate("aircraft",  1166,53   ,satTrail.SatelliteTrail(r=526.1,theta=6.068,width=23.05)),
    Candidate("scattered", 1166,56   ,satTrail.SatelliteTrail(r=368.1,theta=2.369, width=24.41 )), 
    Candidate("aircraft",  1166,58   ,satTrail.SatelliteTrail(r=329.2,theta=2.917, width=22.98 )),
    Candidate("aircraft",  1166,59   ,satTrail.SatelliteTrail(r=666.1,theta=6.063,width=23.17)),
    Candidate("aircraft",  1166,60   ,satTrail.SatelliteTrail(r=1641.3,theta=6.069,width=23.22)),
    Candidate("aircraft",  1166,64   ,satTrail.SatelliteTrail(r=227.1,theta=2.912, width=24.24 )),
    Candidate("aircraft",  1166,65   ,satTrail.SatelliteTrail(r=791.1,theta=6.058, width=22.63 )),
    Candidate("aircraft",  1166,66   ,satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0)),        # small corner
    Candidate("aircraft",  1166,70   ,satTrail.SatelliteTrail(r=245.9,theta=2.904, width=25.30 )),
    Candidate("aircraft",  1166,71   ,satTrail.SatelliteTrail(r=795.8,theta=6.046, width=25.79 )),
    Candidate("aircraft",  1166,72   ,satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0)),        # small corner
    Candidate("moustache", 1166,77   ,satTrail.SatelliteTrail(r=468.5,theta=2.158, width=23.29 )),
    Candidate("satellite", 1166,96   ,satTrail.SatelliteTrail(r=1169.0,theta=6.153,width=17.02 )), 
    Candidate("satellite", 1166,101  ,satTrail.SatelliteTrail(r=542.8,theta=1.147, width=23.95 )),
              

    # 1168 z 
    Candidate("satellite",1168, 4    ,satTrail.SatelliteTrail(r=0.0, theta=0.0 ,width=0.0  )),          # extremely short stub
    Candidate("satellite",1168,10    ,satTrail.SatelliteTrail(r=1071.5,theta=1.397,width=12.40  )),
    Candidate("moustache",1168,15    ,satTrail.SatelliteTrail(r=2349.1,theta=1.128,width=21.86  )),
    Candidate("satellite",1168,16    ,satTrail.SatelliteTrail(r=1208.0,theta=1.405,width=12.08  )),
    Candidate("satellite",1168,23    ,satTrail.SatelliteTrail(r=1547.3,theta=1.410,width=12.80  )),
    Candidate("moustache",1168,29    ,satTrail.SatelliteTrail(r=1682.6,theta=0.783,width=26.18  )),
    Candidate("satellite",1168,29    ,satTrail.SatelliteTrail(r=1962.1,theta=0.981,width=22.84  )),
    Candidate("satellite",1168,31    ,satTrail.SatelliteTrail(r=1875.5,theta=1.419,width=11.87  )),
    Candidate("satellite",1168,39    ,satTrail.SatelliteTrail(r=2188.4,theta=1.424,width=11.59  )),
    Candidate("satellite",1168,47    ,satTrail.SatelliteTrail(r=2492.4,theta=1.431,width=13.12  )),
    Candidate("scattered",1168,50    ,satTrail.SatelliteTrail(r=2135.7,theta=1.590,width=22.39  )),
    Candidate("satellite",1168,55    ,satTrail.SatelliteTrail(r=2786.7,theta=1.439,width=14.29  )),
    Candidate("satellite",1168,63    ,satTrail.SatelliteTrail(r=3070.8,theta=1.441,width=10.96  )),
    Candidate("scattered",1168,66    ,satTrail.SatelliteTrail(r=2236.0,theta=1.576,width=22.89  )),
    Candidate("satellite",1168,71    ,satTrail.SatelliteTrail(r=3339.7,theta=1.446,width=10.46  )),
    Candidate("satellite",1168,78    ,satTrail.SatelliteTrail(r=3599.2,theta=1.452,width=11.81  )),
    Candidate("satellite",1168,84    ,satTrail.SatelliteTrail(r=3843.5,theta=1.456,width=11.31  )),
    Candidate("satellite",1168,90    ,satTrail.SatelliteTrail(r=4072.9,theta=1.468,width=12.79  )),
    Candidate("moustache",1168,101   ,satTrail.SatelliteTrail(r=1136.2,theta=1.230,width=23.51  )),

    
    # 1170 z
    Candidate("scattered",1170, 8    ,satTrail.SatelliteTrail(r=750.5,theta=1.581,width=23.11   )),
    Candidate("moustache",1170,29    ,satTrail.SatelliteTrail(r=2045.5,theta=0.988,width=23.36  )),
    Candidate("scattered",1170,30    ,satTrail.SatelliteTrail(r=577.7,theta=0.023,width=19.39   )),
    Candidate("scattered",1170,46    ,satTrail.SatelliteTrail(r=1048.5,theta=1.574,width=22.36  )),
    Candidate("moustache",1170,69    ,satTrail.SatelliteTrail(r=1023.0,theta=1.590,width=24.52  )),
    Candidate("scattered",1170,83    ,satTrail.SatelliteTrail(r=1196.4,theta=1.586,width=23.31  )),
    Candidate("moustache",1170,89    ,satTrail.SatelliteTrail(r=2512.9,theta=2.027,width=20.38  )),
    Candidate("scattered",1170,89    ,satTrail.SatelliteTrail(r=2685.0,theta=1.582,width=24.57  )),
    Candidate("satellite",1170,90    ,satTrail.SatelliteTrail(r=1127.1,theta=0.252,width=11.36  )),
    Candidate("moustache",1170,96    ,satTrail.SatelliteTrail(r=1339.3,theta=0.646,width=21.22  )),
    Candidate("moustache",1170,101   ,satTrail.SatelliteTrail(r=1210.8,theta=1.261,width=23.42  )),


    # 1180 z
    Candidate("satellite",1180,1     ,satTrail.SatelliteTrail(r=2535.3,theta=1.035,width=12.46  )),
    Candidate("satellite",1180,6     ,satTrail.SatelliteTrail(r=3616.0,theta=1.033,width=10.09  )),
    Candidate("moustache",1180,10    ,satTrail.SatelliteTrail(r=193.2,theta=5.167,width=18.71   )),
    Candidate("satellite",1180,13    ,satTrail.SatelliteTrail(r=871.0,theta=1.029,width=10.99   )),
    Candidate("satellite",1180,19    ,satTrail.SatelliteTrail(r=1970.9,theta=1.023,width=12.81  )),
    Candidate("satellite",1180,16    ,satTrail.SatelliteTrail(r=0.0,theta=0.0,width=0.0  )),          # very faint, but it's there
    Candidate("satellite",1180,17    ,satTrail.SatelliteTrail(r=0.0,theta=0.0,width=0.0  )),          # very faint, but it's there
    Candidate("satellite",1180,18    ,satTrail.SatelliteTrail(r=0.0,theta=0.0,width=0.0  )),          # very faint, but it's there
    Candidate("satellite",1180,25    ,satTrail.SatelliteTrail(r=0.0,theta=0.0,width=0.0  )),          # very faint, but it's there
    Candidate("satellite",1180,26    ,satTrail.SatelliteTrail(r=1619.7,theta=0.185,width=16.67 )),          # very faint, but it's there
    Candidate("satellite",1180,26    ,satTrail.SatelliteTrail(r=3073.0,theta=1.017,width=15.67  )),
    Candidate("scattered",1180,42    ,satTrail.SatelliteTrail(r=3401.9,theta=1.579,width=18.59  )),
    Candidate("satellite",1180,43    ,satTrail.SatelliteTrail(r=1494.7,theta=1.015,width=15.53  )),
    Candidate("satellite",1180,51    ,satTrail.SatelliteTrail(r=2617.8,theta=1.011,width=11.61  )),
    Candidate("satellite",1180,59    ,satTrail.SatelliteTrail(r=3745.1,theta=1.009,width=11.31  )),
    Candidate("scattered",1180,66    ,satTrail.SatelliteTrail(r=3439.1,theta=1.593,width=20.22  )),
    Candidate("scattered",1180,83    ,satTrail.SatelliteTrail(r=1572.5,theta=1.567,width=20.56  )),
    Candidate("moustache",1180,89    ,satTrail.SatelliteTrail(r=2844.6,theta=2.013,width=21.90  )),
    Candidate("moustache",1180,89    ,satTrail.SatelliteTrail(r=2891.2,theta=1.998,width=22.50  )),
    Candidate("moustache",1180,96    ,satTrail.SatelliteTrail(r=1761.1,theta=0.634,width=20.08  )),

    # 1184 z   1 f          96,91,85c,84,78,71,70,62,54
    Candidate("scattered", 1184, 38    ,satTrail.SatelliteTrail(r=1937.9,theta=1.560,width=21.05  )),
    Candidate("meteor",    1184, 54    ,satTrail.SatelliteTrail(r=1281.3,theta=0.819,width=16.11  )),  # trail varies in brightness.  wider than PSF
    Candidate("meteor",    1184, 62    ,satTrail.SatelliteTrail(r=2729.7,theta=0.834,width=13.14  )),
    Candidate("meteor",    1184, 70    ,satTrail.SatelliteTrail(r=0.0,theta=0.840 ,width=0.0   )),  # tiny corner
    Candidate("meteor",    1184, 71    ,satTrail.SatelliteTrail(r=823.2,theta=0.840 ,width=9.81   )),
    Candidate("meteor",    1184, 78    ,satTrail.SatelliteTrail(r=2221.4,theta=0.845,width=18.69  )),
    Candidate("meteor",    1184, 84    ,satTrail.SatelliteTrail(r=3625.3,theta=0.861,width=12.99  )),
    Candidate("meteor",    1184, 85    ,satTrail.SatelliteTrail(r=0.0,theta=0.861,width=0.0  )),  # tiny corner
    Candidate("meteor",    1184, 91    ,satTrail.SatelliteTrail(r=1593.2,theta=0.870,width=20.60  )),
    Candidate("meteor",    1184, 96    ,satTrail.SatelliteTrail(r=2954.7,theta=0.882,width=12.61  )),



    # 1186 z   1 p,d        95,94s     # extremely faint dashes ... will never detect  95,89,83,76,75,67,59,51,42,34,26,19,12,6,1
    Candidate("moustache", 1186,4    ,satTrail.SatelliteTrail(r=1401.1,theta=2.380,width=16.01  )),
    Candidate("moustache", 1186,15   ,satTrail.SatelliteTrail(r=1807.1,theta=0.994,width=17.09  )),
    Candidate("scattered", 1186,22   ,satTrail.SatelliteTrail(r=2152.6,theta=1.587,width=14.57  )),  # maybe a faint moustache
    Candidate("scattered", 1186,30   ,satTrail.SatelliteTrail(r=1315.8,theta=1.578,width=17.88  )),
    Candidate("scattered", 1186,65   ,satTrail.SatelliteTrail(r=693.2,theta=1.583,width=16.67   )),
    Candidate("scattered", 1186,66   ,satTrail.SatelliteTrail(r=1154.1,theta=1.577,width=18.43  )),
    Candidate("scattered", 1186,82   ,satTrail.SatelliteTrail(r=3641.8,theta=1.568,width=16.16  )),
    Candidate("satellite", 1186,94   ,satTrail.SatelliteTrail(r=0.0,theta=0.229,width=0.0    )),    # very short stub
    Candidate("satellite", 1186,95   ,satTrail.SatelliteTrail(r=938.9,theta=0.229,width=8.65    )),
    Candidate("moustache", 1186,100  ,satTrail.SatelliteTrail(r=104.8,theta=5.487,width=13.00   )),

    #1188 z   3 p,b,a      94s,93,86,85,84,78,102  61,60,59,58  15,14,19,18,17,16,100
    Candidate("moustache", 1188,15   ,satTrail.SatelliteTrail(r=2253.9,theta=1.094,width=16.85  )),
    Candidate("aircraft",  1188,14   ,satTrail.SatelliteTrail(r=0.0,theta=6.192,width=0.34   )),
    Candidate("aircraft",  1188,15   ,satTrail.SatelliteTrail(r=0.0,theta=6.192,width=0.34   )),
    Candidate("aircraft",  1188,16   ,satTrail.SatelliteTrail(r=0.0,theta=6.192,width=0.34   )),
    Candidate("aircraft",  1188,17   ,satTrail.SatelliteTrail(r=741.2,theta=6.192,width=20.34   )),
    Candidate("aircraft",  1188,18   ,satTrail.SatelliteTrail(r=1139.9,theta=6.183,width=18.85  )),
    Candidate("aircraft",  1188,19   ,satTrail.SatelliteTrail(r=1554.0,theta=6.173,width=18.57 )),
    Candidate("scattered", 1188,50   ,satTrail.SatelliteTrail(r=1956.4,theta=1.582,width=18.49  )),
    Candidate("satellite", 1188,58   ,satTrail.SatelliteTrail(r=1324.4,theta=6.276,width=8.86   )),
    Candidate("satellite", 1188,59   ,satTrail.SatelliteTrail(r=1342.7,theta=6.277,width=11.13  )),
    Candidate("satellite", 1188,60   ,satTrail.SatelliteTrail(r=1359.9,theta=6.282,width=7.63   )),
    Candidate("satellite", 1188,61   ,satTrail.SatelliteTrail(r=1365.5,theta=6.282,width=8.59  )),
    Candidate("satellite", 1188,78   ,satTrail.SatelliteTrail(r=317.2,theta=0.163,width=8.14    )),
    Candidate("satellite", 1188,84   ,satTrail.SatelliteTrail(r=2427.6,theta=0.171,width=8.47   )),
    Candidate("satellite", 1188,85   ,satTrail.SatelliteTrail(r=1669.0,theta=0.182,width=10.40  )),
    Candidate("satellite", 1188,86   ,satTrail.SatelliteTrail(r=861.4,theta=0.193,width=8.71    )),
    Candidate("moustache", 1188,89   ,satTrail.SatelliteTrail(r=1734.0,theta=2.121,width=17.27  )),
    Candidate("satellite", 1188,93   ,satTrail.SatelliteTrail(r=2083.4,theta=0.205,width=8.46   )),
    Candidate("satellite", 1188,94   ,satTrail.SatelliteTrail(r=0.0,theta=0.205,width=0.0   )),     # tiny stub
    Candidate("moustache", 1188,101  ,satTrail.SatelliteTrail(r=1018.4,theta=1.225,width=18.81  )),
    Candidate("satellite", 1188,102  ,satTrail.SatelliteTrail(r=2816.8,theta=0.155,width=7.19   )),


    # 1190  1190 z   1 b          49s,48,47,46
    Candidate("scattered",1190, 7    ,satTrail.SatelliteTrail(r=1632.1,theta=1.570,width=18.32  )),  # 2 swallow tails
    Candidate("moustache",1190,10    ,satTrail.SatelliteTrail(r=717.0,theta=5.641,width=20.07   )),
    Candidate("moustache",1190,29    ,satTrail.SatelliteTrail(r=2026.7,theta=0.981,width=17.80  )),
    Candidate("satellite",1190,46    ,satTrail.SatelliteTrail(r=424.9,theta=6.274,width=10.57   )),
    Candidate("satellite",1190,47    ,satTrail.SatelliteTrail(r=458.4,theta=6.275,width=10.54   )),
    Candidate("satellite",1190,48    ,satTrail.SatelliteTrail(r=485.2,theta=6.276,width=9.79    )),
    Candidate("satellite",1190,49    ,satTrail.SatelliteTrail(r=485.2,theta=6.276,width=9.79    )),
    Candidate("moustache",1190,69    ,satTrail.SatelliteTrail(r=686.9,theta=1.656,width=17.65   )),
    Candidate("moustache",1190,96    ,satTrail.SatelliteTrail(r=1448.6,theta=0.644,width=16.07  )),
    Candidate("moustache",1190,101   ,satTrail.SatelliteTrail(r=1091.7,theta=1.242,width=18.26  )),
    Candidate("moustache",1190,101   ,satTrail.SatelliteTrail(r=1699.8,theta=0.813,width=17.42  )),


    # 1192 z   1 b          53,52,51,50s
    Candidate("moustache",1192,15    ,satTrail.SatelliteTrail(r=2149.0,theta=0.981,width=15.98  )),
    Candidate("scattered",1192,17    ,satTrail.SatelliteTrail(r=660.9,theta=1.576,width=16.02   )),
    Candidate("scattered",1192,47    ,satTrail.SatelliteTrail(r=1339.5,theta=1.583,width=14.74  )),
    Candidate("satellite",1192,50    ,satTrail.SatelliteTrail(r=1065.0,theta=6.274,width=8.20,  )),  #stub ... near vertical ...split on 2pi
    Candidate("satellite",1192,51    ,satTrail.SatelliteTrail(r=1065.0,theta=6.274,width=8.20,  )),
    Candidate("satellite",1192,52    ,satTrail.SatelliteTrail(r=1096.9,theta=6.275,width=8.12,  )),
    Candidate("satellite",1192,53    ,satTrail.SatelliteTrail(r=1120.2,theta=6.272,width=11.43  )),
    Candidate("scattered",1192,70    ,satTrail.SatelliteTrail(r=3929.6,theta=1.580,width=15.06  )),
    Candidate("scattered",1192,97    ,satTrail.SatelliteTrail(r=2356.5,theta=1.569,width=18.44  )),  #swallow tail ghost

    
    # 1194 z   1 b          57,56,55,54
    Candidate("scattered",1194,34    ,satTrail.SatelliteTrail(r=2691.1,theta=1.583,width=19.31  )),
    Candidate("scattered",1194,37    ,satTrail.SatelliteTrail(r=235.5,theta=2.680,width=16.93  )),  # swallowtail ghost
    Candidate("moustache",1194,38    ,satTrail.SatelliteTrail(r=2461.3,theta=1.571,width=19.20  )),
    Candidate("satellite",1194,54    ,satTrail.SatelliteTrail(r=1682.8,theta=6.271,width=7.70   )),
    Candidate("satellite",1194,55    ,satTrail.SatelliteTrail(r=1732.9,theta=6.274,width=7.54   )),
    Candidate("satellite",1194,56    ,satTrail.SatelliteTrail(r=1771.2,theta=6.274,width=7.50   )),
    Candidate("satellite",1194,57    ,satTrail.SatelliteTrail(r=1805.4,theta=6.273,width=10.91  )),
    Candidate("scattered",1194,68    ,satTrail.SatelliteTrail(r=211.2,theta=1.581,width=18.36  )),
    Candidate("scattered",1194,79    ,satTrail.SatelliteTrail(r=642.3,theta=1.584,width=18.87  )),
    Candidate("scattered",1194,82    ,satTrail.SatelliteTrail(r=1818.3,theta=1.579,width=17.96  )),
    Candidate("scattered",1194,87    ,satTrail.SatelliteTrail(r=208.4,theta=1.582,width=12.02  )),
    Candidate("scattered",1194,94    ,satTrail.SatelliteTrail(r=32.9,theta=2.360,width=17.13  )),
    Candidate("scattered",1194,100   ,satTrail.SatelliteTrail(r=1934.4,theta=1.588,width=23.03  )),

    # 1204 r   1 f      101,15,8,3,2
    Candidate("satellite", 1204,2   ,satTrail.SatelliteTrail(r=2640.9,theta=0.538,width=10.55 )),  
    Candidate("satellite", 1204,3   ,satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0 )),       # there but never successfully measured
    Candidate("satellite", 1204,8   ,satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0 )),       # there but never successfully measured
    Candidate("satellite", 1204,14  ,satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0 )),       # there but never successfully measured
    Candidate("satellite", 1204,101 ,satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0 )),       # there but never successfully measured
    Candidate("moustache",1204,21    ,satTrail.SatelliteTrail(r=3233.7,theta=0.961,width=12.87  )),
    Candidate("moustache",1204,21    ,satTrail.SatelliteTrail(r=3989.1,theta=1.100,width=13.18  )),
    Candidate("moustache",1204,37    ,satTrail.SatelliteTrail(r=1250.9,theta=1.099,width=13.92  )),
    Candidate("moustache",1204,83    ,satTrail.SatelliteTrail(r=2323.6,theta=2.365,width=13.11  )),
    Candidate("moustache",1204,89    ,satTrail.SatelliteTrail(r=1995.0,theta=2.131,width=13.21  )),
    Candidate("moustache",1204,94    ,satTrail.SatelliteTrail(r=482.4,theta=2.560,width=13.06   )),
    Candidate("moustache",1204,94    ,satTrail.SatelliteTrail(r=1592.4,theta=2.424,width=13.05  )),
    Candidate("moustache",1204,94    ,satTrail.SatelliteTrail(r=1643.6,theta=2.403,width=12.97  )),
    Candidate("moustache",1204,94    ,satTrail.SatelliteTrail(r=1695.4,theta=2.381,width=12.51  )),
    Candidate("moustache",1204,95    ,satTrail.SatelliteTrail(r=250.3,theta=5.817,width=13.60   )),
    Candidate("moustache",1204,95    ,satTrail.SatelliteTrail(r=586.9,theta=5.483,width=11.98   )),
    Candidate("moustache",1204,99    ,satTrail.SatelliteTrail(r=357.3,theta=2.523,width=13.24   )),
    Candidate("moustache",1204,101   ,satTrail.SatelliteTrail(r=1039.4,theta=1.248,width=14.23  )),
    Candidate("moustache",1204,103   ,satTrail.SatelliteTrail(r=1827.5,theta=5.486,width=15.99  )),

    # 1206
    Candidate("scattered",1206, 8    ,satTrail.SatelliteTrail(r=561.1,theta=1.571,width=18.70   )),
    Candidate("scattered",1206,46    ,satTrail.SatelliteTrail(r=1511.7,theta=1.548,width=20.26  )),
    Candidate("moustache",1206,54    ,satTrail.SatelliteTrail(r=1808.3,theta=1.561,width=18.60  )),
    Candidate("moustache",1206,69    ,satTrail.SatelliteTrail(r=687.6,theta=1.749,width=18.32   )),
    Candidate("moustache",1206,69    ,satTrail.SatelliteTrail(r=2718.8,theta=1.574,width=16.97  )),
    Candidate("moustache",1206,70    ,satTrail.SatelliteTrail(r=2754.8,theta=1.016,width=23.87  )),
    Candidate("scattered",1206,76    ,satTrail.SatelliteTrail(r=770.0,theta=2.364,width=18.96   )),
    Candidate("moustache",1206,89    ,satTrail.SatelliteTrail(r=350.8,theta=5.688,width=20.04   )),
    Candidate("moustache",1206,89    ,satTrail.SatelliteTrail(r=539.0,theta=2.361,width=19.33   )),
    Candidate("moustache",1206,96    ,satTrail.SatelliteTrail(r=1384.9,theta=0.650,width=20.36  )),
    Candidate("moustache",1206,96    ,satTrail.SatelliteTrail(r=1866.0,theta=0.485,width=21.49  )),
    Candidate("moustache",1206,101    ,satTrail.SatelliteTrail(r=1111.8,theta=1.261,width=18.41  )),
    Candidate("moustache",1206,101    ,satTrail.SatelliteTrail(r=1552.7,theta=0.809,width=18.19  )),
    Candidate("moustache",1206,102    ,satTrail.SatelliteTrail(r=2277.3,theta=1.244,width=24.08  )),

    # 1208
    Candidate("scattered",1208,20    ,satTrail.SatelliteTrail(r=4168.0,theta=1.572,width=12.82  )),
    Candidate("moustache",1208,21    ,satTrail.SatelliteTrail(r=3263.3,theta=0.793,width=11.39  )),
    Candidate("moustache",1208,21    ,satTrail.SatelliteTrail(r=3985.6,theta=1.111,width=11.29  )),
    Candidate("scattered",1208,30    ,satTrail.SatelliteTrail(r=2086.9,theta=1.565,width=12.16  )),
    
    # 1210
    Candidate("scattered",1210, 8    ,satTrail.SatelliteTrail(r=2601.3,theta=1.581,width=17.97  )),
    Candidate("moustache",1210,38    ,satTrail.SatelliteTrail(r=1792.2,theta=1.543,width=19.63  )),
    Candidate("moustache",1210,38    ,satTrail.SatelliteTrail(r=1824.5,theta=1.513,width=19.37  )),
    Candidate("moustache",1210,46    ,satTrail.SatelliteTrail(r=1733.5,theta=1.648,width=18.45  )),
    Candidate("scattered",1210,67    ,satTrail.SatelliteTrail(r=662.9,theta=2.362,width=16.91   )),
    Candidate("scattered",1210,78    ,satTrail.SatelliteTrail(r=1663.7,theta=0.464,width=15.78  )),
    Candidate("moustache",1210,97    ,satTrail.SatelliteTrail(r=185.1,theta=0.049,width=17.13   )),

    # 1216 r   1 f          92s,86,80,73s
    Candidate("satellite", 1216,73  ,satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0 )),       # there but never successfully measured
    Candidate("satellite", 1216,80  ,satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0 )),       # there but never successfully measured
    Candidate("satellite", 1216,86  ,satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0 )),       # there but never successfully measured
    Candidate("satellite", 1216,92  ,satTrail.SatelliteTrail(r=0.0, theta=0.0, width=0.0 )),       # there but never successfully measured

    
    #1236 i   1 b          83s,75,74,66,65,64,56,55
    Candidate("moustache",1236,38    ,satTrail.SatelliteTrail(r=1784.6,theta=1.540,width=17.04  )),
    Candidate("moustache",1236,38    ,satTrail.SatelliteTrail(r=2724.3,theta=1.612,width=17.80  )),
    Candidate("moustache",1236,46    ,satTrail.SatelliteTrail(r=1687.4,theta=1.636,width=18.16  )),
    Candidate("satellite",1236,55    ,satTrail.SatelliteTrail(r=2009.1,theta=0.277,width=8.88   )),
    Candidate("satellite",1236,56    ,satTrail.SatelliteTrail(r=783.6,theta=0.280,width=7.19    )),
    Candidate("satellite",1236,64    ,satTrail.SatelliteTrail(r=2836.1,theta=0.285,width=7.07   )),
    Candidate("satellite",1236,65    ,satTrail.SatelliteTrail(r=1581.4,theta=0.288,width=9.94   )),
    Candidate("satellite",1236,66    ,satTrail.SatelliteTrail(r=380.2,theta=0.286,width=6.77    )),
    Candidate("scattered",1236,67    ,satTrail.SatelliteTrail(r=660.2,theta=2.361,width=17.54   )),
    Candidate("satellite",1236,74    ,satTrail.SatelliteTrail(r=2418.3,theta=0.290,width=7.24   )),
    Candidate("satellite",1236,75    ,satTrail.SatelliteTrail(r=1134.5,theta=0.294,width=8.13   )),
    Candidate("satellite",1236,83    ,satTrail.SatelliteTrail(r=1860.6,theta=0.312,width=11.14  )),

    # 1238 i   
    Candidate("satellite",1238,38    ,satTrail.SatelliteTrail(r=361.0,theta=0.272,width=7.17  )),
    Candidate("satellite",1238,46    ,satTrail.SatelliteTrail(r=0.0,theta=0.272,width=7.17  )),   # very short not detected
    # very faint aircraft running through ccd=15 and others in the column
    

    # 1240 i   2 a,f        96,91,85,86,80,73,74,66,58,59,51,43,44,36,28,29,101  103,77,69c
    Candidate("moustache",1240,15    ,satTrail.SatelliteTrail(r=2751.3,theta=1.097,width=18.40  )),
    Candidate("aircraft",1240,28    ,satTrail.SatelliteTrail(r=2557.6,theta=2.354,width=17.74  )),
    Candidate("aircraft",1240,29    ,satTrail.SatelliteTrail(r=588.8,theta=5.523,width=19.06   )),
    Candidate("moustache",1240,29    ,satTrail.SatelliteTrail(r=745.4,theta=1.283,width=18.16   )),
    Candidate("aircraft",1240,36    ,satTrail.SatelliteTrail(r=892.0,theta=2.388,width=18.71   )),
    Candidate("aircraft",1240,43    ,satTrail.SatelliteTrail(r=2463.3,theta=2.369,width=18.82 )),
    Candidate("aircraft",1240,44    ,satTrail.SatelliteTrail(r=655.8,theta=5.524,width=19.05   )),
    Candidate("aircraft",1240,51    ,satTrail.SatelliteTrail(r=885.6,theta=2.378,width=19.41   )),
    Candidate("aircraft",1240,58    ,satTrail.SatelliteTrail(r=0.8,theta=5.512,width=18.24   )), # as yet undetected
    Candidate("aircraft",1240,59    ,satTrail.SatelliteTrail(r=640.8,theta=5.512,width=18.24   )),
    Candidate("aircraft",1240,66    ,satTrail.SatelliteTrail(r=918.2,theta=2.372,width=20.90   )),
    Candidate("aircraft",1240,73    ,satTrail.SatelliteTrail(r=1.7,theta=5.523,width=20.20   )),  # as yet undetected
    Candidate("aircraft",1240,74    ,satTrail.SatelliteTrail(r=631.7,theta=5.523,width=20.20   )),
    Candidate("aircraft",1240,80    ,satTrail.SatelliteTrail(r=826.0,theta=2.367,width=20.47   )),
    Candidate("scattered",1240,80    ,satTrail.SatelliteTrail(r=2499.8,theta=1.572,width=19.87  )),
    Candidate("aircraft",1240,85    ,satTrail.SatelliteTrail(r=0.8,theta=5.513,width=21.61   )),   # as yet undetected
    Candidate("aircraft",1240,86    ,satTrail.SatelliteTrail(r=706.8,theta=5.513,width=21.61   )),
    Candidate("moustache",1240,89    ,satTrail.SatelliteTrail(r=2263.5,theta=2.138,width=20.15  )),
    Candidate("aircraft",1240,91    ,satTrail.SatelliteTrail(r=932.2,theta=2.362,width=21.81   )),
    Candidate("aircraft",1240,96    ,satTrail.SatelliteTrail(r=566.6,theta=5.498,width=21.53   )),
    Candidate("aircraft",1240,100   ,satTrail.SatelliteTrail(r=613.3,theta=1.974,width=20.22   )),
    Candidate("aircraft",1240,103   ,satTrail.SatelliteTrail(r=1683.9,theta=5.502,width=21.69  )),

    # 1244 i   1 b          3,2,1,0
    Candidate("satellite",1244,0     ,satTrail.SatelliteTrail(r=1064.2,theta=0.033,width=12.54  )),
    Candidate("satellite",1244,1     ,satTrail.SatelliteTrail(r=928.2,theta=0.015,width=9.70    )),
    Candidate("satellite",1244,2     ,satTrail.SatelliteTrail(r=859.0,theta=6.275,width=12.82   )),
    Candidate("satellite",1244,3     ,satTrail.SatelliteTrail(r=885.1,theta=6.259,width=10.82   )),
    Candidate("moustache",1244,84    ,satTrail.SatelliteTrail(r=1082.2,theta=1.042,width=16.93  )),
    
    
    # 1246 i   2 b,b        29,28,27,26  97,96
    Candidate("satellite",1246,26    ,satTrail.SatelliteTrail(r=963.6,theta=0.010,width=13.50   )),
    Candidate("satellite",1246,27    ,satTrail.SatelliteTrail(r=923.3,theta=6.277,width=10.20   )),
    Candidate("satellite",1246,28    ,satTrail.SatelliteTrail(r=931.2,theta=6.269,width=10.96   )),
    Candidate("satellite",1246,29    ,satTrail.SatelliteTrail(r=991.6,theta=6.264,width=6.84    )),
    Candidate("moustache",1246,38    ,satTrail.SatelliteTrail(r=2242.3,theta=1.514,width=18.26  )),
    Candidate("satellite",1246,96    ,satTrail.SatelliteTrail(r=1067.0,theta=0.140,width=12.45  )),
    Candidate("satellite",1246,97    ,satTrail.SatelliteTrail(r=433.9,theta=0.153,width=8.41    )),

    #1248 i   5 b,b,b,f,F  18s,17,16,100  89  29s 48s,49,57,58,59,67,68,69c,77  50,49,48,56,55,62,42,43,44c,36,37
    Candidate("moustache",1248,15    ,satTrail.SatelliteTrail(r=2118.1,theta=0.954,width=17.05  )),
    Candidate("satellite",1248,16    ,satTrail.SatelliteTrail(r=260.3,theta=0.029,width=9.18    )),
    Candidate("satellite",1248,17    ,satTrail.SatelliteTrail(r=142.5,theta=0.021,width=8.90    )),
    Candidate("satellite",1248,18    ,satTrail.SatelliteTrail(r=52.6,theta=0.013,width=9.26     )),
    Candidate("moustache",1248,21    ,satTrail.SatelliteTrail(r=3317.7,theta=0.795,width=17.77  )),
    Candidate("moustache",1248,21    ,satTrail.SatelliteTrail(r=4036.8,theta=1.117,width=17.29  )),
    Candidate("satellite",1248,29    ,satTrail.SatelliteTrail(r=1284.0,theta=0.232,width=7.90   )),
    Candidate("aircraft", 1248,43    ,satTrail.SatelliteTrail(r=605.5,theta=6.013,width=16.36   )),
    Candidate("aircraft", 1248,49    ,satTrail.SatelliteTrail(r=356.3,theta=6.021,width=16.14   )),
    Candidate("satellite",1248,58    ,satTrail.SatelliteTrail(r=1546.3,theta=0.260,width=13.09  )),
    Candidate("satellite",1248,89    ,satTrail.SatelliteTrail(r=331.5,theta=0.048,width=10.81   )),
    Candidate("satellite",1248,100   ,satTrail.SatelliteTrail(r=317.1,theta=0.039,width=7.56    )),

    
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    # Candidate("satellite",    ,satTrail.  )),
    
    #    Candidate("empty",  ,  , satTrail.SatelliteTrail()),
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
    
]

shortCandidates = [
    Candidate("satellite",  242, 95, satTrail.SatelliteTrail(r=1497.8, theta=1.245, width=21.12)),
    Candidate("satellite",   270, 78, satTrail.SatelliteTrail(r=1195.6,theta=5.871,width=14.58)),
    Candidate("aircraft",  1166, 65, satTrail.SatelliteTrail(r= 791.9, theta=6.058, width=22.55)),
    Candidate("satellite",1168,47    ,satTrail.SatelliteTrail(r=2492.4,theta=1.431,width=13.12  )),
    Candidate("meteor",    1184, 78    ,satTrail.SatelliteTrail(r=2221.4,theta=0.845,width=18.69  )),
    Candidate("satellite", 1236, 65, satTrail.SatelliteTrail(r=1580.0, theta=0.286, width= 7.94)),    
    #Candidate("empty",     1236, 50, None),
]
