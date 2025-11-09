from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from photutils.psf import EPSFBuilder, EPSFFitter, extract_stars
from photutils.detection import DAOStarFinder
from astropy.table import Table
from astropy.modeling.fitting import LevMarLSQFitter
import astropy.modeling
import warnings
from photutils.psf import PSFPhotometry, SourceGrouper
from photutils.background import LocalBackground, MMMBackground, MADStdBackgroundRMS
from scipy.ndimage import generic_filter
from astropy.table import vstack
import astropy.wcs 

def epsf_builder(data, starfind_args = {}, epsf_args = {}):
    
    mean, median, std = sigma_clipped_stats(data, sigma = 3)
    
    #can use std when calculating threshold but std value is small and threshold needs to be big ~1000
    starfind = DAOStarFinder(**starfind_args, roundlo = -1.0, roundhi = 1.0, sharplo = 0.2, sharphi = 0.8, exclude_border=True) 
    starcat = starfind(data - median)
    
    size = 25
    hsize = (size - 1) / 2
    x = starcat['xcentroid']  
    y = starcat['ycentroid']  
    mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) & (y > hsize) & (y < (data.shape[0] -1 - hsize))) 
    
    star_cat = Table()
    star_cat['x'] = x[mask]  
    star_cat['y'] = y[mask] 
    
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)  
    data -= median_val  
    
    nddata = NDData(data=data)  
    
    stars = extract_stars(nddata, star_cat, size = 25)
    
    fitter = EPSFFitter(fitter = LevMarLSQFitter())
     
    epsf_builder = EPSFBuilder(**epsf_args, fitter = fitter)  
    epsf, fitted_stars = epsf_builder(stars)

    #modeling ePSF

    y, x = np.mgrid[:epsf.data.shape[0], :epsf.data.shape[1]]
    z = epsf.data
    p_init = astropy.modeling.functional_models.Gaussian2D(x_mean = 25, y_mean = 25, x_stddev = 10., y_stddev = 10.)
    fit_p = LevMarLSQFitter()

    with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, z)

    model_epsf = p(x, y)
    epsf_sigma = np.sqrt(p.x_stddev.value*p.y_stddev.value)/4
    
    return epsf, epsf_sigma, model_epsf

def build_error_image(data, sky_background, gain, readnoise, dark_current):
    '''
    Build an error (1-sigma) image for PSFPhotometry.
    
    Parameters
    ----------
    data : 2D array
        Image in ADU.
    sky_background : 2D or scalar
        Estimated background in ADU (per pixel).
    gain : float
        Gain, e-/ADU.
    readnoise : float
        Read noise in e-.
    dark_current : float
        Dark current in e-/pixel.
    
    Returns
    -------
    error_image : 2D array
        1-sigma uncertainty per pixel, in ADU.
    '''
    # converting the data and sky background to electrons
    data_e = data * gain
    sky_e = sky_background * gain
    
    # 1 sigma error image in electrons
    noise_image = data_e + sky_e + dark_current + readnoise**2
    # clip the negatives
    noise_image = np.clip(noise_image, a_min = 1e-6, a_max = None)
    # convert back to ADU
    error = np.sqrt(noise_image) / gain
    
    return error

def do_photometry(data, scale, epsf, epsf_sigma, error_im = None, finder_args = {}):
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(data)
    bkgstat = MMMBackground()
    localbkg_estimator = LocalBackground(4, 5, bkgstat)
    grouper = SourceGrouper(min_separation = 5)
    finder = DAOStarFinder(threshold = scale * std, fwhm = epsf_sigma * gaussian_sigma_to_fwhm, **finder_args)
    psfphot = PSFPhotometry(psf_model = epsf, fit_shape = 5, finder = finder, grouper = grouper, aperture_radius = 4, localbkg_estimator = localbkg_estimator)

    if error_im is not None:    
        phot_table = psfphot(data, error = error_im)

    else:
        phot_table = psfphot(data)

    return phot_table

def circle_mask(array_shape, x0, y0, radius):

    y, x = np.ogrid[:array_shape[0], :array_shape[1]]
    d = np.sqrt((x - x0)**2 + (y - y0)**2)
    mask = d <= radius
    
    return mask

def nan_median_filter(data, size):
    def nanmedian(values):
        return np.nanmedian(values)
    
    return generic_filter(data, nanmedian, size = size, mode = 'reflect')

def iterative_bkg_removal_and_photometry(catalog, data, wcs, epsf, epsf_sigma, error_im, iters = 1):
    '''
    Iteratively subtract background and run photometry, masking all detected sources permanently.

    Parameters
    ----------
    catalog : Table
        Input catalog with RA/Dec columns (either 'ra'/'dec' or 'RAJ2000'/'DEJ2000').
    data : ndarray
        2D image array.
    wcs : WCS
        WCS object for coordinate transformation.
    epsf : EPSFModel
        Empirical PSF model for photometry.
    epsf_sigma : float
        Sigma parameter for ePSF.
    error_im: 2D array
        The 1 sigma per pixel error image for better flux error 
    iters : int, optional
        Number of iterations to run (default = 1).

    Returns
    -------
    vstack(tables) : combined list of tables 
        If iters == 1, returns a single photometry table.
        Otherwise, returns a combined list of tables from each iteration.
    '''

    #Convert catalog corrds to X/Y coords for the first source mask
    if ('ra' in catalog.keys() and 'dec' in catalog.keys()):
        x, y = wcs.all_world2pix(catalog['ra'], catalog['dec'], 0)
    elif ('RAJ2000' in catalog.keys() and 'DEJ2000' in catalog.keys()):
        x, y = wcs.all_world2pix(catalog['RAJ2000'], catalog['DEJ2000'], 0)
    else:
        raise ValueError('Catalog needs RA and DEC coords')

    init_params = Table([x, y], names=('x_init', 'y_init'))
    
    #removing bad coords
    neg_xy = (init_params['x_init'] < 0) | (init_params['y_init'] < 0)
    init_params = init_params[~neg_xy] 

    #creating the first iteration of the source mask
    mask = np.zeros_like(data, dtype=bool)
    for row in init_params:
        mask |= circle_mask(data.shape, row['x_init'], row['y_init'], radius=2.5)

    #prepping the data (probably not necessary)
    data_copy = data.copy()
    data_copy[np.isnan(data_copy)] = 0
    data_copy[mask] = np.nan

    #first background subtraction
    filtered_image = nan_median_filter(data_copy, size = 20)
    total_bkg = filtered_image.copy() #total background estimation model
    bkgsub_data = data - total_bkg #first background-subtraction

    #first iteration of photometry
    phot_table = do_photometry(bkgsub_data, 3, epsf, epsf_sigma, finder_args = {'sharplo':0.2, 'sharphi':1.4, 'roundlo':-3.0, 'roundhi':3.0})

    if iters == 1:
        return phot_table, bkgsub_data

    #if iters > 1, initialize results + cumulative mask
    tables = [phot_table]

    master_mask = np.zeros_like(data, dtype = bool)
    for row in phot_table:
        master_mask |= circle_mask(data.shape, row['x_init'], row['y_init'], radius = 2.5)

    #iterative background estimation and subtraction into photometry
    for n in range(1, iters):
        #applying the master mask to the background-subtracted data
        bkgsub_data[master_mask] = np.nan

        #creating another estimation of residual background
        filtered_image_loop = nan_median_filter(bkgsub_data, size = 15)
        total_bkg += filtered_image_loop #updating the total background to account for the new estimation 
        bkgsub_data = data - total_bkg #subtracting off the new cumulative background
        bkgsub_data[master_mask] = np.nan #masking previous sources so no repeats/detection of outer PSFs

        #photometry on the iterative background-subtracted image
        phot_table_loop = do_photometry(bkgsub_data, 3, epsf, epsf_sigma, error_im = error_im, finder_args = {'sharplo':0.2, 'sharphi':1.4, 'roundlo':-3.0, 'roundhi':3.0})
        tables.append(phot_table_loop)

        #updating the mask to account for the new sources 
        for row in phot_table_loop:
            master_mask |= circle_mask(data.shape, row['x_init'], row['y_init'], radius = 2.5)

    return vstack(tables)
