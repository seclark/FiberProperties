from __future__ import division, print_function
import numpy as np
from astropy.io import fits

# RHT helper code
import sys 
sys.path.insert(0, '../../RHT')
import RHT_tools

def get_data(verbose = True):
    """
    This can be rewritten for whatever data we're using.
    Currently just grabs some of the SC_241 data from the PRL.
    """
    root = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/"
    data_fn = root + "SC_241.66_28.675.best_20_xyt_w75_s15_t70_galfapixcorr.fits"
    data = fits.getdata(data_fn)
    
    # Print header info
    if verbose == True:
        print(fits.getheader(data_fn))
        
    return data, data_fn

# Load GALFA-HI RHT data
galfa, galfa_fn = get_data()

# Separate into indices and R(theta) arrays
ipoints, jpoints, rthetas, naxis1, naxis2 = RHT_tools.get_RHT_data(galfa_fn)
npoints, nthetas = rthetas.shape
print("There are %d theta bins" %nthetas)

# Number of theta bins to separate RHT data into 
nbins = 10

cube_thetas = np.zeros((naxis1, naxis2, nthetas), np.float_)
cube_thetas[ipoints, jpoints, :] = rthetas
theta_separated_backprojection = np.zeros((naxis1, naxis2, nbins), np.float_)
for i in xrange(nbins):
    theta_separated_backprojection[:, :, i] = np.sum(cube_thetas[:, :, i*nbins:(i+1)*nbins], axis=2)
    


    

    
