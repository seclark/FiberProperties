from __future__ import division, print_function
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage.morphology import grey_erosion, grey_dilation, binary_erosion, binary_dilation
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from reproject import reproject_interp

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

def project_data_into_region():
    
    to_region_fn = "/Volumes/DataDavy/GALFA/SC_241/LAB_corrected_coldens.fits"
    to_region_hdr = fits.getheader(to_region_fn)
    
    allsky_fn = "/Volumes/DataDavy/GALFA/DR2/FullSkyNarrow/GALFA_HI_W_S1011_V-009.2kms.fits"
    allsky_hdr = fits.getheader(allsky_fn)
    allsky_data = fits.getdata(allsky_fn)
     
    new_image, footprint = reproject_interp((allsky_data, allsky_hdr), to_region_hdr) 
    
    return new_image
    
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
    
    gauss_footprint_len = 7
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
    
    for i in xrange(len(ims)):
        axs[i].imshow(ims[i], cmap = "binary")
        axs[i].set_title(titles[i])

def make_cube():
    
    # Let's choose center angle based on average angle for high latitude, fiber-y region
    allQs = np.load("/Volumes/DataDavy/GALFA/SC_241/thetarht_maps/Q_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr.npy")
    allUs = np.load("//Volumes/DataDavy/GALFA/SC_241/thetarht_maps/U_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr.npy")

    mean_angle = np.degrees(np.mod(0.5*np.arctan2(np.nanmean(allUs[:, 4000:]), np.nanmean(allQs[:, 4000:])), np.pi))

    print("Using mean angle {}".format(mean_angle))
    xyv_theta0, hdr = single_theta_velocity_cube(theta_0 = mean_angle, theta_bandwidth = 10)
    
    #fits.writeto("xyv_theta0_"+str(np.round(theta_0))+"_thetabandwidth_"+str(theta_bandwidth)+"_ch"+str(channels[0])+"_to_"+str(channels[-1])+"_new_naxis3.fits", xyv_theta0, hdr)
    
    return xyv_theta0, hdr

def single_theta_velocity_cube(theta_0 = 72, theta_bandwidth = 10, wlen = 75, gaussian_footprint = True):
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
    SC_241_all = fits.getdata("/Volumes/DataDavy/GALFA/SC_241/cleaned/SC_241.66_28.675.best.fits")
    hdr = fits.getheader("/Volumes/DataDavy/GALFA/SC_241/cleaned/SC_241.66_28.675.best.fits")
    
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
        footprint = make_gaussian_footprint(theta_0 = -theta_0, wlen = 7)
    else:
        footprint = make_circular_footprint(radius = 3)
        
    circular_footprint = make_circular_footprint(radius = 2)
    
    # Initialize (x, y, v) cube
    xyv_theta0 = np.zeros((naxis2, naxis1, nchannels), np.float_)
    
    for ch_i, ch_ in enumerate(channels):
        # Grab channel-specific RHT data
        data, data_fn = get_data(ch_, verbose = False)
        ipoints, jpoints, rthetas, naxis1, naxis2 = RHT_tools.get_RHT_data(data_fn)
        
        # Sum relevant thetas
        thetasum_bp = np.zeros((naxis2, naxis1), np.float_)
        thetasum_bp[jpoints, ipoints] = np.nansum(rthetas[:, indx_start:(indx_stop + 1)], axis = 1)
        
        # Erode and dilate
        # Circular erosion
        #eroded_thetasum_bp = erode_data(thetasum_bp, footprint = circular_footprint)
        
        # Gaussian dilation
        #dilated_thetasum_bp = dilate_data(eroded_thetasum_bp, footprint = footprint, structure = footprint)
        
        # Turn into mask
        #mask = np.ones(dilated_thetasum_bp.shape)
        #mask[dilated_thetasum_bp <= 0] = 0
        
        # Try making this a mask first, then binary erosion/dilation
        masked_thetasum_bp = np.ones(thetasum_bp.shape)
        masked_thetasum_bp[thetasum_bp <= 0] = 0
        mask = binary_erosion(masked_thetasum_bp, structure = circular_footprint)
        mask = binary_dilation(mask, structure = footprint)
        
        # Apply mask to relevant velocity data
        realdata_vel_slice = SC_241_all[:, :, ch_]
        realdata_vel_slice[mask == 0] = 0
        
        # Place into channel bin
        xyv_theta0[:, :, ch_i] = realdata_vel_slice
        
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
    
    return xyv_theta0, hdr

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
    



    
