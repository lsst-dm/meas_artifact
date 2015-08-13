#!/usr/bin/env python

import sys, os
import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage
import lsst.meas.algorithms as measAlg
import lsst.meas.artifact as measArt

np.random.seed(42)

class SatelliteTestCase(utilsTests.TestCase):

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
        
        s1 = measArt.SatelliteTrail(r=1769.0,theta=0.017,width=29.88)
        s2 = measArt.SatelliteTrail(r=1769.0,theta=2.0*np.pi-0.001,width=29.88)
        s3 = measArt.SatelliteTrail(r=1769.0,theta=np.pi, width=29.88)
        s4 = measArt.SatelliteTrail(r=1700.0,theta=0.017, width=29.88)

        def compare(trail1, trail2, expect):
            comp = trail1.isNear(trail2, drMax=drMax, dThetaMax=dThetaMax)
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
                isNear = measArt.angleCompare(t, t2, tolerance)
                self.assertEqual(isNear, expected)

                # try adding 2pi and see if it's still correct
                t2 = t + delta + 2.0*np.pi
                isNear = measArt.angleCompare(t, t2, tolerance)
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

        imgX = measArt.separableCrossCorrelate(img, vx, oy)
        imgY = measArt.separableCrossCorrelate(img, ox, vy)

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
        moments = measArt.momentConvolve2d(img, vx, sigma)

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
        constProf = measArt.ConstantProfile(value, width)
        
        flux = 10.0
        sigma = 2.0
        # test both extremes of the DoubleGaussian
        gaussProf  = measArt.DoubleGaussianProfile(flux, sigma, fWing=0.0)
        gaussProf2 = measArt.DoubleGaussianProfile(flux, 0.5*sigma, fWing=1.0)
        
        # vertical trail
        vFake = np.zeros((nx, ny))
        vFake[:,nx//2-1:nx//2+2] += 1.0

        # horizontal trail
        hFake = np.zeros((nx, ny))
        hFake[ny//2-1:ny//2+2,:] += 1.0
        
        for theta, fake in (0.0, vFake), (np.pi/2.0, hFake):
            
            # r,theta  defining the trail
            r = nx//2
            trail = measArt.SatelliteTrail(r, theta)

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

            
    def testShiftOrigin(self):
        """Verify trail origin shift works."""

        # make a trail and shift the origin
        # be sure to shift enough so that we test the
        # reversal in theta when the sign of r changes.
        angle = np.pi/6.0
        trail = measArt.SatelliteTrail(r=1000.0,theta=angle)

        delta = 500.0, 1000.0, 1500.0
        for d in delta:

            # try x
            t = trail.shiftOrigin(d, 0.0)
            rExpected = trail.r - d*np.cos(angle)
            self.assertAlmostEquals(t.r,      np.abs(rExpected))
            tExpected = angle if rExpected >= 0 else angle + np.pi
            self.assertAlmostEquals(t.theta,  tExpected)

            # and round-trip
            t2 = t.shiftOrigin(-d, 0.0)
            self.assertAlmostEquals(t2.r, trail.r)
            self.assertAlmostEquals(t2.theta, trail.theta)
            
            # try y
            t = trail.shiftOrigin(0.0, d)
            rExpected = trail.r - d*np.sin(angle)
            self.assertAlmostEquals(t.r,      np.abs(rExpected))
            tExpected = angle if rExpected >= 0 else angle + np.pi
            self.assertAlmostEquals(t.theta,  tExpected)

            # and round-trip
            t2 = t.shiftOrigin(0.0, -d)
            self.assertAlmostEquals(t2.r, trail.r)
            self.assertAlmostEquals(t2.theta, trail.theta)


    def testHesseForm(self):
        """Verify hesse form conversion"""
        
        n = 10
        nx, ny = 512, 512

        ###########################
        # hesseForm
        # - this just checks that we convert from position angle theta to hesse form theta
        #   correctly (add or subtract pi/2)

        ## try a vertical line
        x0 = (nx//2) * np.ones(n)
        y0 = ny*np.arange(n)
        theta0 = 0.5*np.pi*np.ones(n)

        r, theta = measArt.hesseForm(theta0, x0, y0)

        rResid     = r - nx//2
        thetaResid = theta - 0.0  # expect value is 0.0 ... just showing that.

        self.assertFalse( (np.abs(rResid) > 0.0).any() )
        self.assertFalse( (np.abs(thetaResid) > 0.0).any() )

        ## try a horizontal line
        y0 = (ny//2) * np.ones(n)
        x0 = nx*np.arange(n)
        theta0 = np.zeros(n)

        r, theta = measArt.hesseForm(theta0, x0, y0)

        rResid     = r - ny//2
        thetaResid = theta - np.pi/2.0

        self.assertFalse( (np.abs(rResid) > 1.0e-12).any() )
        self.assertFalse( (np.abs(thetaResid) > 1.0e-12).any() )

        
    def testTwoPiOverlap(self):

        nx, ny = 512, 512
        
        ############################
        # twoPiOverlap
        # - Check that we duplicate points near 0 and 2*pi to wrap around
        #   ... so '2*pi - epsilon' copies to '0 - epsilon', and '0 + epsilon' becomes '2pi + epsilon'

        rang = 0.01*2.0*np.pi
        t = rang*np.arange(100)/99  
        t = np.append(t, 2.0*np.pi - t)
        
        t2, _ = measArt.twoPiOverlap(t, overlapRange=rang)

        tmax = 2.0*np.pi + rang
        tmin = -rang

        self.assertAlmostEqual(tmin, t2.min())
        self.assertAlmostEqual(tmax, t2.max())
        self.assertEqual(2*len(t), len(t2))

    def testThetaAlignment(self):

        np.random.seed(42)
        
        ##############################
        # thetaAlignment
        # - Verify that pair-wise theta comparison works
        #   It should check that a pair of points has the same local theta, and that matches the
        #   theta based on their x,y coords.
        # - To test, generate random points, and then insert a real linear feature.
        #   --> make sure we get back the linear feature points and not the random ones.
        
        n = 500
        nx, ny = 512, 512

        tolerance = 0.1
        # start with random points
        x = nx*np.random.uniform(size=n)
        y = ny*np.random.uniform(size=n)
        t = 2.0*np.pi*np.random.uniform(size=n)

        known = np.zeros(n, dtype=bool)
        
        # append a line
        nLine = 100
        r, theta = 300, 0.4
        trail = measArt.SatelliteTrail(r, theta)
        _x, _y = trail.trace(nx, ny)
        if len(_x) > nLine:
            stride = len(_x)/nLine
            _x = _x[::stride]
            _y = _y[::stride]
        else:
            stride = 1
        nLine = len(_x)

        x = np.append(x, _x)
        y = np.append(y, _y)
        # make sure the thetas pass tolerance... give them a scatter less than the limit.
        t = np.append(t, theta - np.pi/2.0 + 0.1*tolerance*np.random.uniform(size=nLine))

        known = np.append(known, np.ones(nLine, dtype=bool))

        # set ruthless limit=nLine/2  (normally 3 to 5)
        # It means a candidate is accepted only if it has this many statistically unexpected neighbours.
        isCand, newTheta = measArt.thetaAlignment(t, x, y, limit=int(0.5*nLine), tolerance=tolerance)

        # make sure we have the right number of hits
        self.assertEqual(isCand.sum(), nLine)
        # make sure they're for the right objects
        compare = known - isCand
        self.assertEqual(compare.sum(), 0)

    def testImproveCluster(self):
        """Make sure scattered r,theta points convert in improvement algorithm"""

        np.random.seed(42)
        
        nx, ny = 512, 512

        r, theta = 300, 0.4
        trail = measArt.SatelliteTrail(r, theta)
        x, y = trail.trace(nx, ny)
        n = len(x)
        t = theta + 0.05*np.random.normal(size=n)

        niter = 2
        rNew, tNew, _r, _x, _y = measArt.improveCluster(t, x, y)
        for i in range(niter-1):
            rNew, tNew, _r, _x, _y = measArt.improveCluster(tNew, x, y)

        rEst = rNew.mean()
        tEst = tNew.mean()

        # if you need to look at it
        if False:
            r0, theta0 = measArt.hesseForm(t-np.pi/2.0, x, y)
            fig = figure.Figure()
            can = FigCanvas(fig)
            ax = fig.add_subplot(111)
            ax.plot(theta0, r0, 'k.')
            ax.plot(tNew, rNew, '.r')
            ax.plot([tEst], [rEst], 'go')
            fig.savefig("improveCluster.png")

        self.assertLess(np.abs(rEst - r), 1.0)
        self.assertLess(np.abs(tEst - theta), 0.001)


    def testHough(self):
        """ Test binning in r,theta."""

        np.random.seed(42)
        
        nx, ny = 512, 512
        r0, t0 = 300, 0.4
        trail = measArt.SatelliteTrail(r0, t0)
        x, y = trail.trace(nx, ny)
        n = len(x)
        r = r0 + np.random.normal(size=n)
        t = t0 + 0.01*np.random.normal(size=n)

        rMax = (nx**2 + ny**2)**0.5
        result = measArt.hesseBin(r, t, rMax=rMax)
        bin2d, rEdge, tEdge, rs, ts, idx = result

        # make sure we only got one hit
        self.assertEquals(len(rs), 1)
        self.assertEquals(len(ts), 1)
        # and that it has all our points
        self.assertEquals(idx[0].sum(), n)

        # And that it found the right answer
        self.assertClose(rs[0], r0, rtol=1.0e-2)
        self.assertClose(ts[0], t0, rtol=1.0e-2)

        # try the HoughTransform
        hough = measArt.HoughTransform(bins=200, thresh=40, rMax=rMax, maxResid=4.0, nIter=3)

        solutions = hough(t, x, y)

        # these should be better as they run improveCluster() internally
        self.assertEqual(len(solutions), 1)
        self.assertClose(solutions[0].r, r0, rtol=1.0e-3)
        self.assertClose(solutions[0].theta, t0, rtol=1.0e-3)


    def testFindFake(self):
        """Test all the workings together.  Add a fake and find it."""

        np.random.seed(42)

        # make a fake trail in noise ... detect it.
        nx, ny = 512, 512
        r1, t1 = 300, 0.4
        r2, t2 = 200, 2.0*np.pi-0.2
        trail1 = measArt.SatelliteTrail(r1, t1)
        trail2 = measArt.SatelliteTrail(r2, t2)
        
        # create an exposure with a PSF object
        mimg = afwImage.MaskedImageF(nx, ny)
        exposure = afwImage.ExposureF(mimg)
        seeing = 2.0
        kx, ky = 15, 15
        psf = measAlg.DoubleGaussianPsf(kx, ky, seeing/2.35)
        exposure.setPsf(psf)

        # Add two fake trails
        img  = mimg.getImage().getArray()
        flux = 400.0 
        sigma = seeing
        prof = measArt.DoubleGaussianProfile(flux, sigma)
        width = 8*sigma
        trail1.insert(img, prof, width)
        trail2.insert(img, prof, width)

        # add noise on top of the trail
        rms = 20.0
        noise = rms*np.random.normal(size=(nx, ny))
        img += noise
        
        # create a finder and see what we get
        finder          = measArt.SatelliteFinder()
        trailsFind      = finder.getTrails(exposure, widths=[1.0])

        # try a task
        config               = measArt.HoughSatelliteConfig()
        config.narrow.eRange = 0.02
        task                 = measArt.HoughSatelliteTask(config=config)
        trailsTask, runtime  = task.process(exposure)

        # make sure both trails detected by finder and task
        self.assertEqual(len(trailsFind), 2)
        self.assertEqual(len(trailsTask), 2)

        # make sure we found the right ones!
        drMax = 5
        dThetaMax = 0.1
        for trails in trailsFind, trailsTask:
            found1 = trail1.isNear(trails[0], drMax, dThetaMax) or trail1.isNear(trails[1], drMax, dThetaMax)
            found2 = trail2.isNear(trails[0], drMax, dThetaMax) or trail2.isNear(trails[1], drMax, dThetaMax)
            self.assertTrue(found1)
            self.assertTrue(found2)

            
        fig = figure.Figure()
        can = FigCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(img, origin='lower', cmap='gray')
        for trail in trailsTask:
            x1, y1 = trail.trace(nx, ny, offset=10)
            x2, y2 = trail.trace(nx, ny, offset=-10)
            ax.plot(x1, y1, 'r-')
            ax.plot(x2, y2, 'r-')
        fig.savefig("fake.png")
            

    def testPixelSelector(self):
        
        nx, ny = 512, 512
        kwid = 15
        ksig = 90000

        # create a fake image and MomentManager with a vertical line in the middle
        #img = np.random.normal(size=(nx,ny))
        img = np.zeros((nx, ny))
        img[:,nx//2] += 1
        mm = measArt.MomentManager(img, kernelWidth=kwid, kernelSigma=ksig)

        # create a calibration image MomentManager
        cal = np.zeros((kwid,kwid))
        cal[:,kwid//2] += 1
        cmm = measArt.MomentManager(cal, kernelWidth=kwid, kernelSigma=ksig, isCalibration=True)

        # create limits to test
        sumI  = measArt.MomentLimit('sumI',   0.01,  'lower')
        cent  = measArt.MomentLimit('center', 0.01,  'upper')
        ellip = measArt.MomentLimit('ellip',  0.01, 'center')

        # Test the two pixel selectors: basic and p-value
        # --> a basic PixelSelector object (straight cut on values ... pass/fail)
        #     -- inherits from list, so we append our tests to it
        # --> a p-value PixelSelector object (sum log(p) and cut at a specified probability threshold)
        #     -- inherits from PixelSelector base class
        for selectorClass in measArt.PixelSelector, measArt.PValuePixelSelector:
        
            selector = selectorClass(mm, cmm)
            # append the limits ... use (overloaded) append to verify checking valid limit info
            for lim in sumI, cent, ellip:
                selector.append(lim)

            # did we get the right number ... and are they the same pixels?
            goodPixels = selector.getPixels(maxPixels=1000)
            self.assertEqual(ny, goodPixels.sum())
            diff = goodPixels - img
            self.assertFalse(diff.any())
        
        
        
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
       
