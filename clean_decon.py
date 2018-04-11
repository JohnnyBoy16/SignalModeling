# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>, <bojan@bnikolic.co.uk> 
# Initial version August 2010
# http://www.mrao.cam.ac.uk/~bn204/alma/python-clean.html
#
# This file is part of pydeconv. This work is licensed under GNU GPL
# V2 (http://www.gnu.org/licenses/gpl.html)
"""
Clean based deconvolution, using numpy
"""
import numpy as np


def overlapIndices(a1, a2, shiftx):
    
    if shiftx >=0:
        a1xbeg=shiftx
        a2xbeg=0
        a1xend=a1.shape[0]
        a2xend=a1.shape[0]-shiftx
    else:
        a1xbeg=0
        a2xbeg=-shiftx
        a1xend=a1.shape[0]+shiftx
        a2xend=a1.shape[0]

    # if shifty >=0:
    #     a1ybeg=shifty
    #     a2ybeg=0
    #     a1yend=a1.shape[1]
    #     a2yend=a1.shape[1]-shifty
    # else:
    #     a1ybeg=0
    #     a2ybeg=-shifty
    #     a1yend=a1.shape[1]+shifty
    #     a2yend=a1.shape[1]

    # return (a1xbeg, a1xend, a1ybeg, a1yend), (a2xbeg, a2xend, a2ybeg, a2yend)
    return (a1xbeg, a1xend), (a2xbeg, a2xend)

        
def hogbom(dirty, psf, window, gain, thresh, niter):
    """
    Hogbom clean

    :param dirty: The dirty image, i.e., the image to be deconvolved

    :param psf: The point spread-function

    :param window: Regions where clean components are allowed. If
    True, thank all of the dirty image is assumed to be allowed for
    clean components

    :param gain: The "loop gain", i.e., the fraction of the brightest
    pixel that is removed in each iteration

    :param thresh: Cleaning stops when the maximum of the absolute
    deviation of the residual is less than this value

    :param niter: Maximum number of components to make if the
    threshold "thresh" is not hit
    """
    comps = np.zeros(dirty.shape)
    res = np.array(dirty)
    if window is True:
        window = np.ones(dirty.shape, np.bool)

    for i in range(niter):
        # mx, my = np.unravel_index(np.fabs(res[window]).argmax(), res.shape)
        # mval = res[mx, my] * gain
        # comps[mx, my]+=mval
        # a1o, a2o=overlapIndices(dirty, psf,
        #                         mx-dirty.shape[0]/2,
        #                         my-dirty.shape[1]/2)
        # res[a1o[0]:a1o[1],a1o[2]:a1o[3]]-=psf[a2o[0]:a2o[1],a2o[2]:a2o[3]]*mval

        mx = np.fabs(res[window]).argmax()
        mval = res[mx] * gain
        comps[mx] += mval

        a1o, a2o = overlapIndices(dirty, psf, mx-len(dirty)//2)
        res[a1o[0]:a1o[1]] -= psf[a2o[0]:a2o[1]] * mval

        if np.fabs(res).max() < thresh:
            break
    
    return comps, res
