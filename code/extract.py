from __future__ import division, print_function
import numpy as np
from astropy.io import fits

def get_data(verbose = True):
    """
    This can be rewritten for whatever data we're using.
    Currently just grabs some of the SC_241 data from the PRL.
    """
    root = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/"
    data_fn = "SC_241.66_28.675.best_20_xyt_w75_s15_t70_galfapixcorr.fits"
    data = fits.getdata(root + data_fn)
    
    # Print header info
    if verbose == True:
        print(fits.getheader(root + data_fn))
        
    return data
    

    
