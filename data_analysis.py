# Created 08/07/2026
# Author: Joseph Fields
# Description: [[TODO]]

from pathlib import Path
import h5py
import pandas as pd
import numpy as np
import scipy.stats as stats

CHANNEL_EXTRACT_KEYS = ["PlateLFAccX", "PlateLFAccY", "PlateLFAccZ", "PlateHFAccZ", "SpindleAccX", "SpindleAccY", "SpindleAccZ", "Power"]

class DataAnalysis():
    def __init__(self, filename):
        self.data = self.load_file(filename)
        self.data_stats = {}
        pass

    def dereference_data(self, file_handle, dataset):
        """Safely reads data, resolving HDF5 references uniformly as lists if they contain arrays."""
        raw_data = np.array(dataset).T
        
        if raw_data.dtype == object and len(raw_data) > 0:
            resolved_list = []
            for ref in raw_data.flatten():
                if isinstance(ref, h5py.Reference):
                    ref_dataset = file_handle[ref]
                    actual_val = np.array(ref_dataset).T.flatten()
                    
                    # CRITICAL FIX: Always convert the array to a Python list.
                    # This ensures every single row is a list, preventing "mix" errors.
                    resolved_list.append(actual_val.tolist())
                else:
                    # If it's a null/empty reference, provide an empty list to keep types uniform
                    resolved_list.append([])
            return resolved_list
        else:
            # For standard purely numeric/flat columns, return as a flat numpy array
            return raw_data.flatten()       #i dont think this ever happens


    
    def load_file(self, filename):
        # Load the parquet file

        with h5py.File(filename, 'r') as f:
            clean_data = {}
            
            # Find the key that is NOT '#refs#'
            all_keys = list(f.keys()) 
            group_key = all_keys[1]     #second key contains the data

            struct_group = f[group_key]
            
            for channel_key in struct_group.keys():
                channel_data = struct_group[channel_key]
                
                if isinstance(channel_data, h5py.Dataset):
                    clean_data[channel_key] = self.dereference_data(f, channel_data)
        
        return clean_data


    def extract_features(self):
        #mean, standard deviation, root mean squared (RMS), kurtosis, skewness and peak-to-peak; as well as those less commonly used: crest factor, shape factor, impulse factor, margin factor and energy, 
        for channel_name in CHANNEL_EXTRACT_KEYS:
            channel = self.data[channel_name]
            self.data_stats[channel_name] = {}  #need to initialise a dict for each channel
                
            self.data_stats[channel_name]['mean'] = np.mean(channel, axis=1)
            self.data_stats[channel_name]['std'] = np.std(channel, axis=1)
            self.data_stats[channel_name]['RMS'] = np.sqrt(np.mean(np.square(channel), axis=1))
            self.data_stats[channel_name]['kurtosis'] = stats.kurtosis(channel, axis=1)
            self.data_stats[channel_name]['skewness'] = stats.skew(channel, axis=1)
            self.data_stats[channel_name]['p2p'] = np.ptp(channel, axis=1)  # Shortcut for np.max(x) - np.min(x)
            #timeseries
            #crest factor, shape factor, impulse factor, margin factor and energy, which are defined in equation (1)–(5)
            #also spectral kurtosis using STFT - tough to tune
            #could get direction and angle like in Sandvik for vibration? better for force really.

        pass 

if __name__ == "__main__":
    analysis = DataAnalysis('./data/Segmented_Linear_Baseline.mat')
    analysis.extract_features()
    print('test')
    