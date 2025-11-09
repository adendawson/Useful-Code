import numpy as np
import astropy.units as u

def find_fwhm(catalog, image, window_size, pixscale):

    x_0 = np.array(catalog['x_init'])
    y_0 = np.array(catalog['y_init'])
    
    x_int = [int(x) for x in x_0]
    y_int = [int(x) for x in y_0]
    
    cutouts = [image[j-(int(window_size / 2)):j+(int(window_size / 2)), i-(int(window_size / 2)):i+(int(window_size / 2))] for i, j in zip(x_int, y_int)]

    fwhm_all = []

    for i in cutouts:
        yc, xc = np.indices(i.shape)
        mom1_x = (xc * i).sum() / i.sum()
        mom1_y = (yc * i).sum() / i.sum()
    
        mom2_x = ((xc - mom1_x)**2 * i).sum() / i.sum()
        mom2_y = ((yc - mom1_y)**2 * i).sum() / i.sum()
    
        sigma_x, sigma_y = mom2_x**0.5, mom2_y**0.5
        sigma_to_fwhm = np.sqrt(8*np.log(2))
        fwhm_x, fwhm_y = sigma_x*sigma_to_fwhm, sigma_y*sigma_to_fwhm
    
        fwhm_x_arcsec = fwhm_x*u.pixel * pixscale
        fwhm_y_arcsec = fwhm_y*u.pixel * pixscale
    
        fwhm = ((fwhm_x_arcsec*fwhm_y_arcsec)**0.5).value

        fwhm_all.append(fwhm)

    return fwhm_all
