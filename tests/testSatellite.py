#!/usr/bin/env python

import sys, os
import unittest
import numpy as np
import lsst.utils.tests as utilsTests
import satellite.satelliteTrail as satTrail
import satellite.satelliteUtils as satUtil

class SatelliteTestCase(unittest.TestCase):

    def setUp(self):
        pass
    def tearDown(self):
        pass
    
    def testIsNear(self):
        """Check isNear() method at 2pi wrap.

        This makes a call to angleCompare() [see next test], so it should fine,
        but it checks the inner workings of SatelliteTrail.
        """

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
        

    def testAngleCompare(self):
        """Verify that angle comparisons are correct with 2pi wrapping."""

        n = 20
        thetas = 2.0*np.pi*np.arange(n)/n

        tolerance = 0.1
        for t in thetas:
            # try values within and outside tolerance
            for delta,expected in (0.01, True), (0.2, False):
                t2 = t + delta
                isNear = satUtil.angleCompare(t, t2, tolerance)
                self.assertEqual(isNear, expected)

                # try adding 2pi and see if it's still correct
                t2 = t + delta + 2.0*np.pi
                isNear = satUtil.angleCompare(t, t2, tolerance)
                self.assertEqual(isNear, expected)

                
    def testSeparableCrossCorrelate(self):
        """Test that separableCrossCorrelate does the right thing."""
        
        nx, ny = 11, 11

        vx = np.arange(nx) - nx//2
        vy = np.arange(ny) - ny//2
        xx, yy = np.meshgrid(vx, vy)

        ox = np.ones(nx)
        oy = np.ones(ny)
        
        img = np.zeros((nx, ny))
        img[ny//2,nx//2] = 1.0

        imgX = satUtil.separableCrossCorrelate(img, vx, oy)
        imgY = satUtil.separableCrossCorrelate(img, ox, vy)

        # cross-correlating will flip the symmetry wrt meshgrid, but should otherwise be the same
        testX = np.abs(imgX - xx[::,::-1]) > 0.0
        testY = np.abs(imgY - yy[::-1,::]) > 0.0
        self.assertFalse(testX.any())
        self.assertFalse(testY.any())


    def testMomentConvolve2d(self):
        """Test momentConvolve2d() routine.

        The code works the same way as separableCrossCorrelate, but it's separate code
        (there are advantages in speed to doing all the correlations together).

        This is test is therefore similar to testSeparableCrossCorrelate(), but checks
        the higher moments.
        """

        nx, ny = 11, 11

        vx = np.arange(nx) - nx//2
        vy = np.arange(ny) - ny//2
        xx, yy = np.meshgrid(vx, vy)

        ox = np.ones(nx)
        oy = np.ones(ny)

        # to test, just put a delta-function in the middle, cross-correlation
        # should be the reverse image of the kernel
        img = np.zeros((nx, ny))
        img[ny//2,nx//2] = 1.0

        sigma = 1.0e99
        moments = satUtil.momentConvolve2d(img, vx, sigma)

        testX  = np.abs(moments.ix  - xx[::,::-1])    > 0.0
        testY  = np.abs(moments.iy  - yy[::-1,::])    > 0.0
        testXX = np.abs(moments.ixx - xx[::,::-1]**2) > 0.0
        testYY = np.abs(moments.iyy - yy[::,::-1]**2) > 0.0
        testXY = np.abs(moments.ixy - xx[::,::-1]*yy[::-1,::]) > 0.0
        
        self.assertFalse(testX.any())
        self.assertFalse(testY.any())
        self.assertFalse(testXX.any())
        self.assertFalse(testYY.any())
        self.assertFalse(testXY.any())



    def testTrails(self):
        """Test insert and measure"""


        ######################################
        # start with a small constant profile trail
        #
        # - add it and see if we can measure it
        ######################################
        
        nx, ny = 31, 31
        
        value = 1.0 # the value to insert
        width = 2.1 # the width of the trail
        constProf = satTrail.ConstantProfile(value, width)
        
        flux = 10.0
        sigma = 2.0
        # test both extremes of the DoubleGaussian
        gaussProf  = satTrail.DoubleGaussianProfile(flux, sigma, fWing=0.0)
        gaussProf2 = satTrail.DoubleGaussianProfile(flux, 0.5*sigma, fWing=1.0)
        
        # vertical trail
        vFake = np.zeros((nx, ny))
        vFake[:,nx//2-1:nx//2+2] += 1.0

        # horizontal trail
        hFake = np.zeros((nx, ny))
        hFake[ny//2-1:ny//2+2,:] += 1.0
        
        for theta, fake in (0.0, vFake), (np.pi/2.0, hFake):
            
            # r,theta  defining the trail
            r = nx//2
            trail = satTrail.SatelliteTrail(r, theta)

            # check the length
            leng = trail.length(nx, ny)
            self.assertEquals(leng, nx)
            
            measWidth = 10.0*width
            ###################################
            # insert the trail
            profiles = [
                (constProf, nx*(int(width) + 1.0)),
                (gaussProf,  nx*flux),
                (gaussProf2, nx*flux)
                ]
            
            for prof, expectedFlux in profiles:
                
                img = np.zeros((nx, ny))
                trail.insert(img, prof, measWidth)

                # measure it
                measFlux = trail.measure(img, widthIn=measWidth)

                # centroid offset should be zero
                self.assertAlmostEquals(trail.center, 0.0)
                # flux should be same as returned value
                self.assertEquals(trail.flux, measFlux)
                self.assertAlmostEquals(trail.flux, expectedFlux, 4)

            # ConstProf should match the fake exactly
            img = np.zeros((nx, ny))
            trail.insert(img, constProf, width+2.0)
            test = np.abs(img - fake) > 0.0
            self.assertFalse(test.any())

        
            
        
#################################################################
# Test suite boiler plate
#################################################################
def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SatelliteTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
       
