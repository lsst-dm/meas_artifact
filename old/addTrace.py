#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import lssttools.functions as func

class Gauss2d(object):

    def __init__(self, r, theta, sigma, f_wing=0.1):
        self.r = r
        self.theta = theta
        self.sigma = sigma
        self.vx = np.cos(theta)
        self.vy = np.sin(theta)
        
        self.f_core = 1.0 - f_wing
        self.f_wing = f_wing
        if self.f_wing < 0.0 or self.f_wing > 1.0:
            raise ValueError("f_wing must be in the range 0.0 .. 1.0")
        
    def __call__(self, x, y):

        dot = x*self.vx + y*self.vy
        offset = np.abs(dot - self.r)
        wy,wx = np.where(offset < 5.0*self.sigma)
        out = np.zeros(x.shape)
        A1 = 1.0/(2.0*np.pi*self.sigma**2)
        g1 = np.exp(-offset[wy,wx]**2/(2.0*self.sigma**2))
        A2 = 1.0/(2.0*np.pi*(2.0*self.sigma)**2)
        g2 = np.exp(-offset[wy,wx]**2/(2.0*(2.0*self.sigma)**2))
        out[wy,wx] += self.f_core*g1 + self.f_wing*g2
        return out
        
        
def addTrace(img, r, theta, flux, sigma=2.0):

    xx, yy = np.meshgrid(np.arange(img.shape[1]).astype(int), np.arange(img.shape[0]).astype(int))
    g = Gauss2d(r, theta, sigma)
    return img + flux*g(xx, yy)

    
if __name__ == '__main__':
    
    img = np.zeros((32,32))
    r = 16
    theta = 0.5
    img2 = addTrace(img, r, theta)

    plt.imshow(img2)
    plt.savefig('g.png')
