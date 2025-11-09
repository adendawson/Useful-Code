import numpy as np
from scipy.signal import savgol_filter

class spectrum_analysis:
    def __init__(self, wavelength, flux):
        self.wavelength = np.array(wavelength)
        self.flux = np.array(flux)
        self.telluric_removed = False
        self.segments = None
        self.absorption_removed = False
        self.emission_removed = False
        self.continuum_fits_dict = {}
        self.rms_results = {}
        self.best_fit = {}

    def remove_telluric(self, transmission, transmission_wl, airmass_telluric, airmass_target):
        target_transmission = transmission**(airmass_target / airmass_telluric)
        wl_corrected_tt = np.interp(self.wavelength, transmission_wl, target_transmission)
        self.flux = self.flux / wl_corrected_tt
        self.telluric_removed = True
        return self.flux
        
    def segment_spectrum(self, window_size):
        n_points = len(self.wavelength)
        n_segments = int(round(n_points / window_size, 1))
        self.segments = []
        for i in range(n_segments):
            start = i * window_size
            end = min((i + 1) * window_size, n_points)
            wl_segment = self.wavelength[start:end]
            flux_segment = self.flux[start:end]
            self.segments.append((wl_segment, flux_segment))
        return self.segments

    def remove_absorption(self, prominence_values = np.arange(0, 25, 0.1), distance = 1):
        if self.segments is None:
            raise RuntimeError('Run segment_spectrum() before removing absorption features.')

        cleaned_segments = {}
        prominence_values = [round(i, 2) for i in prominence_values]
        for p in prominence_values:
            cleaned_segments[p] = []
            for wl_segment, flux_segment in self.segments:
                cleaned_flux = minimum_remover(wl_segment, flux_segment, prominence = p, distance = distance, mode = 'absorption')
                cleaned_segments[p].append((wl_segment, cleaned_flux))
        self.absorption_removed = True
        self.cleaned_absorption = cleaned_segments
        return cleaned_segments

    def remove_emission(self, prominence_values = np.arange(0, 25, 0.1), distance = 1):
        if self.absorption_removed == False:
            raise RuntimeError('You must call remove_absorption() before remove_emission().')

        cleaned_segments = {}
        prominence_values = [round(i, 2) for i in prominence_values]
        for p in prominence_values:
            cleaned_segments[p] = []
            for wl_segment, flux_segment in self.cleaned_absorption[p]:
                cleaned_flux = minimum_remover(wl_segment, flux_segment, prominence = p, distance = distance, mode = 'emission')
                cleaned_segments[p].append((wl_segment, cleaned_flux))
        self.emission_removed = True
        self.cleaned_emission = cleaned_segments
        return cleaned_segments

    def best_continuum_fit(self, window_length, polyorder):

        if self.emission_removed == True:    
            for p, segments in self.cleaned_emission.items():
                continuum_fits = []
                for wl_segment, flux_segment in segments:
                    continuum = savgol_filter(flux_segment, window_length = window_length, polyorder = polyorder)
                    continuum_fits.append((wl_segment, continuum))
                self.continuum_fits_dict[p] = continuum_fits
        
            rms_results = {}
            for p, segments in self.continuum_fits_dict.items():
                rms_vals = []
                for (wl_segment, continuum), (_, orig) in zip(segments, self.cleaned_emission[p]):
                    rms = np.sqrt(np.mean((orig - continuum) ** 2))
                    rms_vals.append(rms)
                rms_results[p] = rms_vals
            self.rms_results = rms_results
        
            min_prominence = [np.argmin([self.rms_results[p][i] for p in self.rms_results.keys()]) + 1 for i in range(len(self.segments))]
        
            wl_arr = []
            flux_arr = []
            prominence_values = [round(i, 2) for i in np.arange(0, 25, 0.1)]
            min_p_vals = [prominence_values[i] for i in min_p_vals]
            for i, prominence in zip(range(len(self.segments)), min_prominence):
                best_wl_segment, best_flux_segment = self.cleaned_emission[prominence][i]
                wl_arr.append(best_wl_segment)
                flux_arr.append(best_flux_segment)
        
            self.final_wl = np.concatenate((wl_arr))
            self.final_flux = np.concatenate((flux_arr))
            
            return self.final_wl, self.final_flux

        else:    
            for p, segments in self.cleaned_absorption.items():
                continuum_fits = []
                for wl_segment, flux_segment in segments:
                    continuum = savgol_filter(flux_segment, window_length = window_length, polyorder = polyorder)
                    continuum_fits.append((wl_segment, continuum))
                self.continuum_fits_dict[p] = continuum_fits
        
            rms_results = {}
            for p, segments in self.continuum_fits_dict.items():
                rms_vals = []
                for (wl_segment, continuum), (_, orig) in zip(segments, self.cleaned_absorption[p]):
                    rms = np.sqrt(np.mean((orig - continuum) ** 2))
                    rms_vals.append(rms)
                rms_results[p] = rms_vals
            self.rms_results = rms_results
        
            min_prominence = [np.argmin([self.rms_results[p][i] for p in self.rms_results.keys()]) + 1 for i in range(len(self.segments))]
        
            wl_arr = []
            flux_arr = []
            prominence_values = [round(i, 2) for i in np.arange(0, 25, 0.1)]
            min_p_vals = [prominence_values[i] for i in min_p_vals]
            for i, prominence in zip(range(len(self.segments)), min_prominence):
                best_wl_segment, best_flux_segment = self.cleaned_absorption[prominence][i]
                wl_arr.append(best_wl_segment)
                flux_arr.append(best_flux_segment)
        
            self.final_wl = np.concatenate((wl_arr))
            self.final_flux = np.concatenate((flux_arr))
            
            return self.final_wl, self.final_flux
