from __future__ import division, print_function
import numpy as np
from astropy.io import fits
from scipy.ndimage.morphology import grey_erosion, grey_dilation
import matplotlib.pyplot as plt

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

def bin_data_by_theta(nbins = 10):
    """
    Places RHT data into cube binned by theta. 
    
    Input:
    nbins :: number of theta bins to separate RHT data into (default = 10)
    
    Output:
    theta_separated_backprojection :: 3D array, dimensions (naxis1, naxis2, nbins).
                                   :: contains backprojection summed into nbins chunks.
    """
    
    # Load GALFA-HI RHT data
    galfa, galfa_fn = get_data()

    # Separate into indices and R(theta) arrays
    ipoints, jpoints, rthetas, naxis1, naxis2 = RHT_tools.get_RHT_data(galfa_fn)
    npoints, nthetas = rthetas.shape
    print("There are %d theta bins" %nthetas)

    cube_thetas = np.zeros((naxis2, naxis1, nthetas), np.float_)
    cube_thetas[jpoints, ipoints, :] = rthetas
    theta_separated_backprojection = np.zeros((naxis2, naxis1, nbins), np.float_)
    for i in xrange(nbins):
        theta_separated_backprojection[:, :, i] = np.sum(cube_thetas[:, :, i*nbins:(i+1)*nbins], axis=2)
        
    return theta_separated_backprojection

def erode_data(data, footprint = None):
    
    if footprint is not None:
        eroded_data = grey_erosion(data, footprint = footprint)
    else:
        eroded_data = grey_erosion(data, size=(10, 10))
    
    return eroded_data
    
def dilate_data(data, footprint = None):
    
    if footprint is not None:
        dilated_data = grey_dilation(data, footprint = footprint)
    else:
        dilated_data = grey_dilation(data, size=(10, 10))
    
    return dilated_data
    
def make_footprint(radius = 10):
    """
    Make a footprint of a given radius (in pixels) for use in erosion/dilation.
    """
    
    fp = np.zeros((2*radius+1, 2*radius+1), np.float_)
    mnvals = np.indices(fp.shape)
    mvals = mnvals[:, :][0] # These are the y points
    nvals = mnvals[:, :][1] # These are the x points
    rads = np.zeros(fp.shape, np.float_)
    rads = np.sqrt((mvals-radius)**2 + (nvals-radius)**2)
    
    fp[rads < radius] = 1
    
    return fp

def erode_dilate_example(nbins = 10, footprint_radius = 3):
    # Create an array of backprojections binned by theta    
    theta_separated_backprojection = bin_data_by_theta(nbins = nbins)

    # Scale to [0, 1]
    theta_separated_backprojection = theta_separated_backprojection/np.nanmax(theta_separated_backprojection)

    # Create a circular footprint for use in erosion / dilation.
    footprint = make_footprint(footprint_radius = 3)

    # Erode and dilate the backprojection to rid us of single pixels and/or small isolated objects.
    # As a test, we are working with thetabin = 4
    eroded_theta4 = erode_data(theta_separated_backprojection[:, :, 4], footprint = footprint)
    dilated_theta4 = dilate_data(eroded_theta4, footprint = footprint)

    # Simple plotting -- show original, eroded, and dilated data for thetabin = 4.
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.imshow(theta_separated_backprojection[:, :, 4])
    ax2.imshow(eroded_theta4)
    ax3.imshow(dilated_theta4)

    ax1.set_title("Original backprojection")
    ax2.set_title("Eroded backprojection")
    ax3.set_title("Dilated backprojection")

    #plt.savefig("marytest.png")

    

    
