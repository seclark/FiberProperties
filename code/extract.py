from __future__ import division, print_function
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage.morphology import grey_erosion, grey_dilation, binary_erosion, binary_dilation
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from reproject import reproject_interp
import copy

# RHT helper code
import sys 
sys.path.insert(0, '../../RHT')
import RHT_tools

def get_data(chan = 20, verbose = True):
    """
    This can be rewritten for whatever data we're using.
    Currently just grabs some of the SC_241 data from the PRL.
    """
    
    #root = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/"
    root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/"
    data_fn = root + "SC_241.66_28.675.best_"+str(chan)+"_xyt_w75_s15_t70_galfapixcorr.fits"
    data = fits.getdata(data_fn)
    
    # Print header info
    if verbose == True:
        print(fits.getheader(data_fn))
        
    return data, data_fn

def bin_data_by_theta(data_fn, nbins = 10):
    """
    Places RHT data into cube binned by theta. 
    
    Input:
    data_fn :: fits filename for RHT data
    nbins :: number of theta bins to separate RHT data into (default = 10)
    
    Output:
    theta_separated_backprojection :: 3D array, dimensions (naxis1, naxis2, nbins).
                                   :: contains backprojection summed into nbins chunks.
    """

    # Separate into indices and R(theta) arrays
    ipoints, jpoints, rthetas, naxis1, naxis2 = RHT_tools.get_RHT_data(data_fn)
    npoints, nthetas = rthetas.shape
    print("There are %d theta bins" %nthetas)

    cube_thetas = np.zeros((naxis2, naxis1, nthetas), np.float_)
    cube_thetas[jpoints, ipoints, :] = rthetas
    theta_separated_backprojection = np.zeros((naxis2, naxis1, nbins), np.float_)
    for i in xrange(nbins):
        theta_separated_backprojection[:, :, i] = np.sum(cube_thetas[:, :, i*nbins:(i+1)*nbins], axis=2)
        
    return theta_separated_backprojection
    
def coadd_by_angle():
    """
    Adds theta-binned velocity slices
    """
    # Define channels
    channels = [19, 20]
    
    # Number of desired theta bins
    nthetabins = 10
    
    # Initial data
    data, data_fn = get_data(channels[0], verbose = False)
    theta_separated_backprojection = bin_data_by_theta(data_fn, nbins = nthetabins)
    
    # Add to existing theta_separated_backprojection
    for chan in channels[1:]:
        data, data_fn = get_data(chan, verbose = False)
        theta_separated_backprojection += bin_data_by_theta(data_fn, nbins = nthetabins)
        
def radecs_to_xy(w, ras, decs):
    
    #Transformation
    radec = zip(ras, decs)
    xy = w.wcs_world2pix(radec, 1)
    
    xs = xy[:,0]
    ys = xy[:,1]
    
    return xs, ys
    
def xys_to_radec(w, xs, ys):
    
    #Transformation
    xy = zip(xs, ys)
    radec = w.wcs_pix2world(xy, 1)
    
    ras = radec[:,0]
    decs = radec[:,1]
    
    return ras, decs

def project_data_into_region(from_data_fn, to_region = "SC_241"):
    
    if to_region == "SC_241":
        to_region_fn = "/Volumes/DataDavy/GALFA/SC_241/LAB_corrected_coldens.fits"
        to_region_hdr = fits.getheader(to_region_fn)
    
    from_data_hdr = fits.getheader(from_data_fn)
    from_data_data = fits.getdata(from_data_fn)
     
    new_image, footprint = reproject_interp((from_data_data, from_data_hdr), to_region_hdr) 
    
    return new_image
    
def get_velocity_from_fits(fits_fn, kms = True):
    """
    Returns velocities of fits cube slices (assumes they are in NAXIS3)
    """

    hdr = fits.getheader(fits_fn)
    
    vels = np.zeros(hdr["NAXIS3"])
    
    for i in xrange(len(vels)):
        vels[i] = hdr["CRVAL3"] + (i)*hdr["CDELT3"]
        
    if kms is True:
        vels = vels/1000.0
        
    return vels
    
def make_projected_cube():
    import glob

    #1000 - 1047 will be velocities -017.3kms to +017.3kms
    start_num = 1000
    nvels = 48
    to_region_hdr = fits.getheader("/Volumes/DataDavy/GALFA/SC_241/LAB_corrected_coldens.fits")
    projected_narrow_data = np.zeros((to_region_hdr["NAXIS2"], to_region_hdr["NAXIS1"], nvels), np.float_)
    
    for i in xrange(nvels):
        num = start_num + i
        print("Analyzing channel number {}".format(num))
        allsky_fns = glob.glob("/Volumes/DataDavy/GALFA/DR2/FullSkyNarrow/GALFA_HI_W_S"+str(num)+"_*.fits")
        
        projected_narrow_data[:, :, i] = project_data_into_region(allsky_fns[0], to_region = "SC_241")
        
    # Whole cube velocities from Yong
    vels_wide = np.loadtxt("GALFAHI_VLSR_W.txt")
        
    hdr = copy.copy(to_region_hdr)
    hdr["NAXIS3"] = nvels
    hdr["CRVAL3"] = vels_wide[start_num]*1000 # starting velocity in m/s
    hdr["CRDELT3"] = (vels_wide[1] - vels_wide[0])*1000 # velocity delta in m/s
    
    fits.writeto("GALFA_HI_W_projected_SC_241.fits", projected_narrow_data, hdr)
            
    return projected_narrow_data
    
def test_different_erosion_dilations(wlen = 75, theta_0 = 72, theta_bandwidth = 10):

    # Get thetas for given window length
    thets = RHT_tools.get_thets(wlen)
    
    # Get index of theta bin that is closest to theta_0
    indx_0 = (np.abs(thets - np.radians(theta_0))).argmin()
    
    # Get index of beginning and ending thetas that are approximately theta_bandwidth centered on theta_0
    indx_start = (np.abs(thets - (thets[indx_0] - np.radians(theta_bandwidth/2.0)))).argmin()
    indx_stop = (np.abs(thets - (thets[indx_0] + np.radians(theta_bandwidth/2.0)))).argmin()

    data, data_fn = get_data(20, verbose = False)
    ipoints, jpoints, rthetas, naxis1, naxis2 = RHT_tools.get_RHT_data(data_fn)
    
    # Sum relevant thetas
    thetasum_bp = np.zeros((naxis2, naxis1), np.float_)
    thetasum_bp[jpoints, ipoints] = np.nansum(rthetas[:, indx_start:(indx_stop + 1)], axis = 1)
    
    # Only look at subset
    thetasum_bp = thetasum_bp[:, 4000:]

    # Try making this a mask first, then binary erosion/dilation
    masked_thetasum_bp = np.ones(thetasum_bp.shape)
    masked_thetasum_bp[thetasum_bp <= 0] = 0
    
    gauss_footprint_len = 5
    circ_footprint_rad = 2
    footprint = make_gaussian_footprint(theta_0 = -theta_0, wlen = gauss_footprint_len)
    circular_footprint = make_circular_footprint(radius = circ_footprint_rad)
    
    # circular binary erosion then gaussian binary dilation
    cbe_then_gbd = binary_erosion(masked_thetasum_bp, structure = circular_footprint)
    cbe_then_gbd = binary_dilation(cbe_then_gbd, structure = footprint)
    
    # gaussian binary dilation then circular binary erosion
    gbd_then_cbe = binary_dilation(masked_thetasum_bp, structure = footprint)
    gbd_then_cbe = binary_erosion(gbd_then_cbe, structure = circular_footprint)
    
    # circular binary erosion then circular binary dilation
    cbe_then_cbd = binary_dilation(masked_thetasum_bp, structure = circular_footprint)
    cbe_then_cbd = binary_erosion(cbe_then_cbd, structure = circular_footprint)
    
    # circular binary dilation then circular binary erosion
    cbd_then_cbe = binary_erosion(masked_thetasum_bp, structure = circular_footprint)
    cbd_then_cbe = binary_dilation(cbd_then_cbe, structure = circular_footprint)
    
    # gaussian binary dilation then gaussian binary erosion
    gbd_then_gbe = binary_dilation(masked_thetasum_bp, structure = footprint)
    gbd_then_gbe = binary_erosion(gbd_then_gbe, structure = footprint)
    
    # gaussian binary erosion then gaussian binary dilation
    gbe_then_gbd = binary_erosion(masked_thetasum_bp, structure = footprint)
    gbe_then_gbd = binary_dilation(gbe_then_gbd, structure = footprint)
    
    fig = plt.figure(facecolor = "white")
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    
    ims = [cbe_then_gbd, gbd_then_cbe, cbe_then_cbd, cbd_then_cbe, gbd_then_gbe, gbe_then_gbd]
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    titles = ["Circ Erosion, Gauss Dilation", "Gauss Dilation, Circ Erosion", "Circ Erosion, Circ Dilation", "Circ Dilation, Circ Erosion", "Gauss Dilation, Gauss Erosion", "Gauss Erosion, Gauss Dilation"]
    
    plt.suptitle("Binary Erosion and Dilation, Circular radius = {}, Gaussian Kernel Length = {}".format(circ_footprint_rad, gauss_footprint_len))
    
    for i in xrange(len(ims)):
        axs[i].imshow(ims[i], cmap = "binary")
        axs[i].set_title(titles[i])

def make_cube():
    
    # Let's choose center angle based on average angle for high latitude, fiber-y region
    allQs = np.load("/Volumes/DataDavy/GALFA/SC_241/thetarht_maps/Q_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr.npy")
    allUs = np.load("/Volumes/DataDavy/GALFA/SC_241/thetarht_maps/U_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr.npy")

    mean_angle = np.degrees(np.mod(0.5*np.arctan2(np.nanmean(allUs[:, 4000:]), np.nanmean(allQs[:, 4000:])), np.pi))

    print("Using mean angle {}".format(mean_angle))
    xyv_theta0, hdr = single_theta_velocity_cube(theta_0 = mean_angle, theta_bandwidth = 10)
    
    #fits.writeto("xyv_theta0_"+str(np.round(theta_0))+"_thetabandwidth_"+str(theta_bandwidth)+"_ch"+str(channels[0])+"_to_"+str(channels[-1])+"_new_naxis3.fits", xyv_theta0, hdr)
    
    return xyv_theta0, hdr

def single_theta_velocity_cube(theta_0 = 72, theta_bandwidth = 10, wlen = 75, smooth_radius = 60, gaussian_footprint = True, gauss_footprint_len = 5, circular_footprint_radius = 3, binary_erode_dilate = True):
    """
    Creates cube of Backprojection(x, y, v | theta_0)
    where dimensions are x, y, and velocity
    and each velocity slice contains the backprojection for that velocity, at a single theta.
    
    theta_0         :: approximate center theta value (in degrees). 
                       Actual value will be closest bin in xyt data.
    theta_bandwidth :: approximate width of theta range desired (in degrees)
    wlen            :: RHT window length
    
    """
    
    # Read in all SC_241 *original* data
    SC_241_original_fn = "/Volumes/DataDavy/GALFA/SC_241/cleaned/SC_241.66_28.675.best.fits"
    SC_241_all = fits.getdata(SC_241_original_fn)
    hdr = fits.getheader(SC_241_original_fn)
    
    # Velocity data should be third axis
    SC_241_all = SC_241_all.swapaxes(0, 2)
    SC_241_all = SC_241_all.swapaxes(0, 1)
    naxis2, naxis1, nchannels_total = SC_241_all.shape
    
    # Get thetas for given window length
    thets = RHT_tools.get_thets(wlen)
    
    # Get index of theta bin that is closest to theta_0
    indx_0 = (np.abs(thets - np.radians(theta_0))).argmin()
    
    # Get index of beginning and ending thetas that are approximately theta_bandwidth centered on theta_0
    indx_start = (np.abs(thets - (thets[indx_0] - np.radians(theta_bandwidth/2.0)))).argmin()
    indx_stop = (np.abs(thets - (thets[indx_0] + np.radians(theta_bandwidth/2.0)))).argmin()
    
    print("Actual theta range will be {} to {} degrees".format(np.degrees(thets[indx_start]), np.degrees(thets[indx_stop])))
    print("Theta indices will be {} to {}".format(indx_start, indx_stop))
    
    # Define velocity channels
    channels = [20]#[16, 17, 18, 19, 20, 21, 22, 23, 24]
    nchannels = len(channels)
    
    # Create a circular footprint for use in erosion / dilation.
    if gaussian_footprint is True:
        footprint = make_gaussian_footprint(theta_0 = theta_0, wlen = gauss_footprint_len)
        
        # Mathematical definition of kernel flips y-axis
        footprint = footprint[::-1, :]
    else:
        footprint = make_circular_footprint(radius = circular_footprint_radius)
    
    # Initialize (x, y, v) cube
    xyv_theta0 = np.zeros((naxis2, naxis1, nchannels), np.float_)
    
    for ch_i, ch_ in enumerate(channels):
        # Grab channel-specific RHT data
        data, data_fn = get_data(ch_, verbose = False)
        ipoints, jpoints, rthetas, naxis1, naxis2 = RHT_tools.get_RHT_data(data_fn)
        
        # Sum relevant thetas
        thetasum_bp = np.zeros((naxis2, naxis1), np.float_)
        thetasum_bp[jpoints, ipoints] = np.nansum(rthetas[:, indx_start:(indx_stop + 1)], axis = 1)
        
        if binary_erode_dilate is False:
            # Erode and dilate
            eroded_thetasum_bp = erode_data(thetasum_bp, footprint = footprint, structure = footprint)
            dilated_thetasum_bp = dilate_data(eroded_thetasum_bp, footprint = footprint, structure = footprint)
        
            # Turn into mask
            mask = np.ones(dilated_thetasum_bp.shape)
            mask[dilated_thetasum_bp <= 0] = 0
        
        else:
            # Try making this a mask first, then binary erosion/dilation
            masked_thetasum_bp = np.ones(thetasum_bp.shape)
            masked_thetasum_bp[thetasum_bp <= 0] = 0
            mask = binary_erosion(masked_thetasum_bp, structure = footprint)
            mask = binary_dilation(mask, structure = footprint)
        
        # Apply background subtraction to velocity slice.
        background_subtracted_data = background_subtract(mask, SC_241_all[:, :, ch_], smooth_radius = smooth_radius, plotresults = False)
        
        # Place into channel bin
        xyv_theta0[:, :, ch_i] = background_subtracted_data
        
    #return xyv_theta0
    hdr["CHSTART"] = channels[0]
    hdr["CHSTOP"] = channels[-1]
    hdr["NAXIS3"] = len(channels)
    hdr["THETA0"] = theta_0
    hdr["THETAB"] = theta_bandwidth
    hdr["CRPIX3"] = hdr["CRPIX3"] - channels[0]
    
    # Deal with python fits axis ordering
    xyv_theta0 = xyv_theta0.swapaxes(0, 2)
    xyv_theta0 = xyv_theta0.swapaxes(1, 2)
    
    #fits.writeto("xyv_theta0_"+str(theta_0)+"_thetabandwidth_"+str(theta_bandwidth)+"_ch"+str(channels[0])+"_to_"+str(channels[-1])+"_new_naxis3.fits", xyv_theta0, hdr)
    
    return xyv_theta0, hdr, mask
    
def background_subtract(mask, data, smooth_radius = 60, plotsteps = False, plotresults = True):
    """
    Background subtraction
    """
    
    # Reverse-mask to get background
    background_data = copy.copy(data)
    background_data[mask == 1] = None
    
    circ_footprint = make_circular_footprint(radius = smooth_radius)
    smooth_background_data = smooth_overnans(background_data, filter = "median", footprint = circ_footprint)
    
    smooth_background_data[np.isnan(smooth_background_data)] = 0
    background_subtracted_data = data - smooth_background_data
    
    thresholded_masked_data = copy.copy(background_subtracted_data)
    thresholded_masked_data[background_subtracted_data <= 0] = 0
    thresholded_masked_data[mask == 0] = 0
    
    naively_masked_data = copy.copy(data)
    naively_masked_data[mask == 0] = 0
    
    if plotsteps is True:
        fig = plt.figure(facecolor = "white")
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.imshow(data)
        ax1.set_title("data")
        ax2.imshow(smooth_background_data)
        ax2.set_title("smooth background data")
        ax3.imshow(background_subtracted_data)
        ax3.set_title("background subtracted data")
        ax4.imshow(thresholded_masked_data)
        ax4.set_title("thresholded masked data")
        
    if plotresults is True:
        fig = plt.figure(facecolor = "white")
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        im1 = ax1.imshow(thresholded_masked_data[:, 4000:], cmap = "binary")
        ax1.set_title("Background subtracted masked data")    
        plt.colorbar(im1, ax = ax1)
        
        im2 = ax2.imshow(naively_masked_data[:, 4000:], cmap = "binary")
        ax2.set_title("Naively masked data")    
        plt.colorbar(im2, ax = ax2)
    
    return thresholded_masked_data
    
    
def smooth_overnans(map, sig = 15, filter = "median", footprint = None):

    """
    Takes map with nans, etc set to 0
    """
    
    mask = np.ones(map.shape, np.float_)
    mask[np.isnan(map)] = 0
    
    map_zeroed = copy.copy(map)
    map_zeroed[mask == 0] = 0
    
    if filter == "gauss":
        blurred_map = ndimage.gaussian_filter(map_zeroed, sigma=sig)
        blurred_mask = ndimage.gaussian_filter(mask, sigma=sig)
        
    if filter == "median":
        blurred_map = ndimage.filters.median_filter(map_zeroed, footprint = footprint)
        blurred_mask = ndimage.filters.median_filter(mask, footprint = footprint)
    
    map = blurred_map / blurred_mask
  
    return map

def erode_data(data, footprint = None, structure = None):
    
    if footprint is not None:
        eroded_data = grey_erosion(data, footprint = footprint, structure = structure)
    else:
        eroded_data = grey_erosion(data, size=(10, 10))
    
    return eroded_data
    
def dilate_data(data, footprint = None, structure = None):
    
    if footprint is not None:
        dilated_data = grey_dilation(data, footprint = footprint, structure = structure)
    else:
        dilated_data = grey_dilation(data, size=(10, 10))
    
    return dilated_data

def make_gaussian_footprint(theta_0 = 0, wlen = 50):
    """
    Make a gaussian footprint 
    """
    
    fp = np.zeros((wlen, wlen), np.float_)
    mnvals = np.indices(fp.shape)
    mvals = mnvals[:, :][0] # These are the y points
    nvals = mnvals[:, :][1] # These are the x points
    
    initial_cov = np.asarray([[10, 0], [0, 1]])
    psi = np.radians(theta_0)
    rotation_matrix = np.asarray([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    covariance_matrix = np.dot(rotation_matrix, np.dot(initial_cov, rotation_matrix.T))
    
    var = multivariate_normal(mean=[wlen/2.0, wlen/2.0], cov=covariance_matrix)
    
    indices = [(y, x) for y, x in zip(mvals.ravel(), nvals.ravel())]

    fp = var.pdf(indices)
    fp = fp.reshape(wlen, wlen)
    
    return fp

def plot_gaussian_footprints():
    """
    Plot various gaussian footprints for an example
    """
    
    fig = plt.figure(facecolor = "white")
    ax1 = fig.add_subplot(151)
    ax2 = fig.add_subplot(152)
    ax3 = fig.add_subplot(153)
    ax4 = fig.add_subplot(154)
    ax5 = fig.add_subplot(155)
    
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    angles = [0, 30, 70, 90, 110]
    
    wlen = 50
    
    for i, ang in enumerate(angles):
        fp = make_gaussian_footprint(theta_0 = ang, wlen = wlen)
        axes[i].imshow(fp, cmap = "cubehelix")
        axes[i].set_title(r"$"+str(ang)+"^\circ$")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
def make_circular_footprint(radius = 10):
    """
    Make a circular footprint of a given radius (in pixels) for use in erosion/dilation.
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
    
    # Grab default RHT data
    galfa, galfa_fn = get_data()

    # Create an array of backprojections binned by theta    
    theta_separated_backprojection = bin_data_by_theta(nbins = nbins)

    # Scale to [0, 1]
    theta_separated_backprojection = theta_separated_backprojection/np.nanmax(theta_separated_backprojection)

    # Create a circular footprint for use in erosion / dilation.
    footprint = make_circular_footprint(radius = 3)

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
    
# This is where the stuff that gets executed when you run "python extract.py" goes.
"""
if __name__ == "__main__":
    
    # Center theta (degrees)
    theta_0 = 72
    
    # Width of theta window around theta_0 (degrees)
    theta_bandwidth = 10
    
    # RHT window length (pixels)
    wlen = 75
    
    # Radius of median filter for background subtraction (pixels)
    smooth_radius = 60
    
    # If True, erode and dilate data using gaussian footprint. Otherwise, circular.
    gaussian_footprint = True
    
    # Length of one side of gaussian footprint
    gauss_footprint_len = 5
    
    # Radius of circular footprint
    circular_footprint_radius = 3
    
    # If True, binary erosion/dilation. If False, grey erosion/dilation.
    binary_erode_dilate = True
    
    
    # Run code
    xyv_theta0, hdr, mask = single_theta_velocity_cube(theta_0 = theta_0, theta_bandwidth = theta_bandwidth, wlen = wlen,
                               smooth_radius = smooth_radius, gaussian_footprint = gaussian_footprint, 
                               gauss_footprint_len = gauss_footprint_len, circular_footprint_radius = circular_footprint_radius, 
                               binary_erode_dilate = binary_erode_dilate)

    # Save output cube to fits file
    xyv_theta0_fn = "xyv_theta0_"+str(theta_0)+"_thetabandwidth_"+str(theta_bandwidth)+"_ch"+str(channels[0])+"_to_"+str(channels[-1])+".fits"
    fits.writeto(xyv_theta0_fn, xyv_theta0, hdr)
"""

