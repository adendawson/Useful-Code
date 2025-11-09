from astropy.coordinates import SkyCoord
import astropy.units as u

def coords_deg_to_hms(ra, dec):
    c = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    ra_str = f'{int(c.ra.hms[0])} {int(c.ra.hms[1])} {round(c.ra.hms[2], 2)}'
    dec_str = f'{int(c.dec.dms[0])} {int(c.dec.dms[1])} {round(c.dec.dms[2], 2)}'
    
    return ra_str, dec_str

def coords_hms_to_deg(ra, dec):
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    ra_deg, dec_deg = c.ra.value, c.dec.value
    
    return ra_deg, dec_deg
