#!/usr/bin/env python

import sys, os
import unittest
import numpy as np
import lsst.utils.tests as utilsTests
import satelliteTrail as satTrail

class SatelliteTestCase(unittest.TestCase):

    def setUp(self):
        pass
    def tearDown(self):
        pass
    
    # Check isNear() method at 2pi wrap
    def testIsNear(self):

        drMax     = 50.0
        dThetaMax = 0.15
        
        s1 = satTrail.SatelliteTrail(r=1769.0,theta=0.017,width=29.88)
        s2 = satTrail.SatelliteTrail(r=1769.0,theta=2.0*np.pi-0.001,width=29.88)
        s3 = satTrail.SatelliteTrail(r=1769.0,theta=np.pi, width=29.88)
        s4 = satTrail.SatelliteTrail(r=1700.0,theta=0.017, width=29.88)

        def compare(trail1, trail2, expect):
            comp = trail1.isNear(trail2, drMax=drMax, dThetaMax=dThetaMax)
            print comp, expect
            if expect:
                self.assertTrue(comp)
            else:
                self.assertFalse(comp)
                
        compare(s1, s1, True)
        compare(s2, s2, True)
        compare(s1, s2, True)
        compare(s2, s1, True)
        compare(s3, s1, False)
        compare(s3, s2, False)
        compare(s4, s1, False)
        compare(s4, s2, False)
        


#################################################################
# Test suite boiler plate
#################################################################
def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SatelliteTestCase)
    #suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
       
