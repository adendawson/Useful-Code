import numpy as np
import pandas
from scipy.signal import savgol_filter, find_peaks
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from scipy.interpolate import interp1d
from collections import defaultdict
from astropy.stats import mad_std
from astroquery.nist import Nist
import math
from scipy.optimize import curve_fit

class SpectrumAnalysis:
    def __init__(self, wavelength, flux):
        self.wavelength = np.asarray(wavelength, dtype = float)
        self.flux = np.asarray(flux, dtype = float)

        if self.wavelength.shape != self.flux.shape:
            raise ValueError(
                f'wavelength and flux must have the same shape, got {self.wavelength.shape} and {self.flux.shape}.')
        if self.wavelength.ndim != 1:
            raise ValueError('wavelength and flux must be 1D arrays.')
        
        self.telluric_removed = False
        self.barycentric_corrected = False
        self.segments = None
        self.pipeline_steps = []
        self.cleaned_flux = None
        self.cleaned_wavelength = None
        self._last_cleaned_segments = None
        self.ran_nist = False

#correcting the spectrum
    def remove_telluric(self, transmission, transmission_wl, airmass_telluric, airmass_target, pixel_shift = 0):
        
        target_transmission = transmission**(airmass_target / airmass_telluric)
        wl_corrected_tt = np.interp(self.wavelength, transmission_wl, target_transmission)

        #negative pixel_shift is left, positive is right
        if pixel_shift != 0:
            shifted_tt = np.roll(wl_corrected_tt, pixel_shift)
            if pixel_shift > 0:
                shifted_tt[:pixel_shift] = wl_corrected_tt[0]
            else:
                shifted_tt[pixel_shift:] = wl_corrected_tt[-1]
            
            wl_corrected_tt = shifted_tt
            
        self.flux = self.flux / wl_corrected_tt
        self.telluric_removed = True
        
        return self.flux, wl_corrected_tt

    def barycentric_correction(self, observatory_lat, observatory_lon, observatory_alt, obs_time, target_ra, target_dec):
    
        location = EarthLocation(lat = observatory_lat, lon = observatory_lon, height = observatory_alt)
        t = Time(obs_time, scale = 'utc')
        target = SkyCoord(ra = target_ra * u.deg, dec = target_dec * u.deg, frame = 'icrs')
    
        barycentric_correction = target.radial_velocity_correction(obstime = t, location = location)
    
        c = 299792.458  # km/s
        v = barycentric_correction.to(u.km / u.s).value
    
        correction_factor = np.sqrt((1 + v/c) / (1 - v/c))
        
        self.wavelength = self.wavelength / correction_factor
    
        self.barycentric_corrected = True
    
        print(f'Barycentric velocity: {v:.4f} km/s')
        print(f'Correction factor: {correction_factor:.8f}')
        print(f'Wavelength shift at midpoint: {self.wavelength[len(self.wavelength)//2] * (1/correction_factor - 1):.4f} Å')
    
        #return interp_flux

#segmenting the spectrum
        
    def segment_spectrum(self, window_size):
        
        n_points = len(self.wavelength)
        n_segments = math.ceil(n_points / window_size)
        
        self.segments = []
        for i in range(n_segments):
            start = i * window_size
            end = min((i + 1) * window_size, n_points)
            wl_segment = self.wavelength[start:end]
            flux_segment = self.flux[start:end]
            self.segments.append((wl_segment, flux_segment))
        return self.segments

#fitting the continuum

    def make_clean_spectrum(self, segment_idx, prominence_values, mode, chain = False):

        flux_array = []
        wl_array = []
    
        for i, (idx, p) in enumerate(zip(segment_idx, prominence_values)):
            wl_segment = self.segments[idx][0]
    
            if chain:
                if self._last_cleaned_segments is None:
                    raise RuntimeError('chain = True requires a previous call to make_clean_spectrum.')
                flux_segment = self._last_cleaned_segments[i][1]
            else:
                flux_segment = self.segments[idx][1]
    
            cleaned_segment = minimum_remover(wl_segment, flux_segment, p, mode = mode, distance = 1)
            flux_array.append(cleaned_segment)
            wl_array.append(wl_segment)
    
        self._last_cleaned_segments = [(wl_array[i], flux_array[i]) for i in range(len(segment_idx))]
    
        flux_array = np.concatenate(flux_array)
        wl_array = np.concatenate(wl_array)
    
        self.cleaned_flux = flux_array
        self.cleaned_wavelength = wl_array
    
        return wl_array, flux_array

    def fit_continuum_sigma_clip(self, window_length, polyorder, sigma_low = 2.0, sigma_high = 3.0, n_iter = 5, local_window = 100):
        
        flux_input = self.cleaned_flux if self.cleaned_flux is not None else self.flux
        
        # Interpolate over any nans before iterating
        nan_mask = np.isnan(flux_input)
        if nan_mask.any():
            x = np.arange(len(flux_input))
            flux_input = np.interp(x, x[~nan_mask], flux_input[~nan_mask])
    
        mask = np.ones(len(flux_input), dtype = bool)
        flux_work = flux_input.copy()
        pad = window_length // 2
        half_win = local_window // 2
    
        for _ in range(n_iter):
            flux_padded = np.pad(flux_work, pad, mode = 'reflect')
            continuum_padded = savgol_filter(flux_padded, window_length, polyorder)
            continuum = continuum_padded[pad:-pad]
    
            residuals = flux_work - continuum
            local_std = np.array([
                np.std(residuals[max(0, i - half_win) : min(len(residuals), i + half_win)])
                for i in range(len(residuals))
            ])
    
            mask = (residuals > -sigma_low * local_std) & (residuals < sigma_high * local_std)
            flux_work = np.where(mask, flux_input, continuum)
    
        self.final_flux = flux_work
        self.final_wl = self.cleaned_wavelength if self.cleaned_flux is not None else self.wavelength
    
        return self.final_wl, self.final_flux

    def preview_segment_removal(self, segment_idx, prominence_values, mode, window_length, polyorder, xlim = None, ylim = None):

        if self.segments is None:
            raise RuntimeError('Run segment_spectrum() first.')
    
        raw_wl, raw_flux = self.segments[segment_idx]
    
        fig, ax = pl.subplots(figsize = (10, 6))
        ax.plot(raw_wl, raw_flux, color = 'black', lw = 1.0, label = 'Raw segment', zorder = 5)
    
        for p in prominence_values:
            if mode == 'absorption':
                cleaned = minimum_remover(raw_wl, raw_flux, prominence = p, distance = 1, mode = 'absorption')
            elif mode == 'emission':
                cleaned = minimum_remover(raw_wl, raw_flux, prominence = p, distance = 1, mode = 'emission')
            elif mode == 'both':
                cleaned = minimum_remover(raw_wl, raw_flux, prominence = p, distance = 1, mode = 'absorption')
                cleaned = minimum_remover(raw_wl, cleaned,  prominence = p, distance = 1, mode = 'emission')
            else:
                raise ValueError('mode must be "absorption", "emission", or "both".')
    
            continuum = savgol_filter(cleaned, window_length = window_length, polyorder = polyorder)
            ax.plot(raw_wl, cleaned, alpha = 0.55, label = f'Cleaned (p = {p})')
            ax.plot(raw_wl, continuum, linestyle = '--', label = f'Continuum (p = {p})')
    
        ax.set_xlabel('Wavelength (Angstrom)')
        ax.set_ylabel('Arbitrary Flux')
        ax.set_title(f'Preview — Segment {segment_idx}, mode="{mode}"')
        ax.legend(loc = 'best')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        pl.tight_layout()

    def normalize_spectrum(self, window_length = 25, polyorder = 3):
        if window_length % 2 == 0:
            window_length += 1
        if window_length > len(self.final_flux):
            raise ValueError(f"window_length ({window_length}) exceeds the length of final_flux ({len(self.final_flux)}). Use a smaller window.")
        
        continuum_cleaned = savgol_filter(self.final_flux, window_length, polyorder)
        continuum_interp = np.interp(self.wavelength, self.final_wl, continuum_cleaned)
        
        self.normalized_spectrum = (self.flux / continuum_interp) - 1
        return self.normalized_spectrum

#spectral line identification
    def run_nist(self, linenames = None):

        if linenames is None:
            linenames = ['H I', 'He I', 'He II',
                         'Ca I', 'Ca II', 
                         'Mg I', 'Mg II',
                         'Na I', 
                         'Fe I', 'Fe II', 'Fe III',
                         'Si I', 'Si II', 
                         'C I', 'C II', 'C III', 'C IV',
                         'N I', 'N II', 'N III', 
                         'O I', 'O II',
                         'K I', 
                         'Ti I', 'Ti II', 
                         'Th I']
    
        wmin, wmax = np.min(self.wavelength), np.max(self.wavelength)
    
        all_lines = Nist.query(wmin * u.AA, wmax * u.AA, linename = linenames, wavelength_type = 'vacuum', output_order = 'wavelength')
    
        wl_rest = np.array(all_lines['Ritz'], dtype=float)
        good = np.isfinite(wl_rest)
        filtered_lines = all_lines[good]
    
        line_db = defaultdict(list)
        for row in filtered_lines:
            line_db[row['Spectrum']].append(row)
    
        self.line_library = line_db
        self.linenames = linenames

        all_wl = []
        for element in linenames:
            df = pandas.DataFrame.from_dict(line_db[element])
            wave = [df[2][i] for i in range(len(df))]
            all_wl.append(wave)
        
        all_wl = np.array(np.concatenate((all_wl)))
        self.all_nist_wl = all_wl[~np.isnan(all_wl)]
        self.ran_nist = True
    
        return self.line_library, self.linenames, self.all_nist_wl

    def find_radial_velocity(self, sigma_factor = 3, c = 3e5, tolerance = 10):
        std = mad_std(self.normalized_spectrum, ignore_nan = True)
        threshold = sigma_factor * std
        emission_peaks, _ = find_peaks(self.normalized_spectrum, height = threshold, distance = 1)
        absorption_peaks, _ = find_peaks(-self.normalized_spectrum, height = threshold, distance = 1)
        all_peaks = np.concatenate((emission_peaks, absorption_peaks))
        all_peaks = np.sort(all_peaks)
    
        fitted_peak_wl = []
        fitted_peak_wl_err = []
        for peak_idx in all_peaks:
            if peak_idx < 15:
                arr1 = self.normalized_spectrum[:peak_idx][::-1]
            else:
                arr1 = self.normalized_spectrum[peak_idx - 15:peak_idx][::-1]
            arr2 = self.normalized_spectrum[peak_idx:peak_idx + 16]
        
            is_absorption = peak_idx in absorption_peaks
        
            if is_absorption:
                max_1 = first_local_max(arr1)
                max_2 = first_local_max(arr2)
        
                idx1 = 0 if peak_idx - max_1 == 0 else peak_idx - (max_1 + 1)
                idx2 = peak_idx + max_2
        
            else:
                min_1 = first_local_min(arr1)
                min_2 = first_local_min(arr2)
                
                idx1 = 0 if peak_idx - min_1 == 0 else peak_idx - (min_1 + 1)
                idx2 = peak_idx + min_2
        
            flux_for_fit = -self.normalized_spectrum if is_absorption else self.normalized_spectrum
        
            line_flux = flux_for_fit[idx1:idx2+1]
            line_wl = self.wavelength[idx1:idx2+1]
        
            wave_disp = np.median(np.diff(self.wavelength))
        
            p0 = [self.wavelength[peak_idx], flux_for_fit[peak_idx], 0.5, 1.0]   # initial guesses for x0, A, sigma, gamma
            bounds = ([self.wavelength[peak_idx] - (5*wave_disp), flux_for_fit[peak_idx] - 10, 0.01, 0.01],
                      [self.wavelength[peak_idx] + (5*wave_disp), flux_for_fit[peak_idx] + 10,  5.0,  5.0])
            
            popt, pcov = curve_fit(voigt, line_wl, line_flux, p0 = p0, bounds = bounds, nan_policy = 'omit')
            perr = np.sqrt(np.diag(pcov))
        
            peak_wl = popt[0]
            peak_wl_err = perr[0]
        
            fitted_peak_wl.append(peak_wl)
            fitted_peak_wl_err.append(peak_wl_err)
        
        fitted_peak_wl = np.asarray(fitted_peak_wl)
        fitted_peak_wl_err = np.asarray(fitted_peak_wl_err)
        
        emission_peaks_wl = np.array(fitted_peak_wl)[:, np.newaxis] 
        all_wl = np.array(self.all_nist_wl).flatten()                                        
            
        diff = np.abs(emission_peaks_wl - all_wl)                                  
        
        best_match_idx = np.argmin(diff, axis = 1)         
        best_mask = diff[np.arange(len(emission_peaks_wl)), best_match_idx] <= tolerance
        
        obs_wl = emission_peaks_wl[best_mask, 0]
        rest_wl = all_wl[best_match_idx[best_mask]]
        z = (obs_wl - rest_wl) / rest_wl
        velocities = c * ((1 + z)**2 - 1) / ((1 + z)**2 + 1)
        self.rv_result_tbl = pandas.DataFrame({'obs_wavelength': obs_wl, 'rest_wavelength': rest_wl, 'delta_lambda': obs_wl - rest_wl, 'velocity_km_s': velocities})
        self.velocities = velocities
        
        return self.velocities, self.rv_result_tbl, fitted_peak_wl