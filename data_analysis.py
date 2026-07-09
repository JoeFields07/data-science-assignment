# Created 08/07/2026
# Author: Joseph Fields
# Description: [[TODO]]

from pathlib import Path
import h5py
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CHANNEL_KEYS = ["PlateLFAccX", "PlateLFAccY", "PlateLFAccZ", "PlateHFAccZ", "SpindleAccX", "SpindleAccY", "SpindleAccZ", "Power"]
FEATURE_KEYS = ['mean', 'std', 'RMS', 'kurtosis', 'skewness', 'p2p', 'crest_factor', 'shape_factor', 'impulse_factor', 'margin_factor', 'energy']

class DataAnalysis():
    def __init__(self, filename, verbose=True):
        self.verbose = verbose
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
        print(f"Loading {filename}") if self.verbose else 0
        with h5py.File(filename, 'r') as f:
            clean_data = {}
            
            # Find the key that is NOT '#refs#'
            all_keys = list(f.keys())     
            data_group = f[all_keys[1]]   #second key contains the data
            
            for channel_key in data_group.keys():     #could replace with CHANNEL_KEYS
                channel_data = data_group[channel_key]
                
                if isinstance(channel_data, h5py.Dataset):  #could remove
                    clean_data[channel_key] = self.dereference_data(f, channel_data)
        print("Data imported successfully") if self.verbose else 0
        return clean_data


    def remove_data(self):
        self.data = {}      #remove the raw data to save memory once features have been extracted
        print("Removed raw data") if self.verbose else 0
        pass


    def extract_features(self):
        #mean, standard deviation, root mean squared (RMS), kurtosis, skewness and peak-to-peak; 
        #as well as those less commonly used: crest factor, shape factor, impulse factor, margin factor and energy, 
        print("Starting feature extraction") if self.verbose else 0
        for channel_name in CHANNEL_KEYS:
            channel = self.data[channel_name]
            self.data_stats[channel_name] = {}      #need to initialise a dict for each channel
            
            x_squared = np.square(channel)          #pre-calculate components to save time
            x_max = np.max(channel, axis=1)
            x_mean_abs = np.mean(np.abs(channel), axis=1)

            self.data_stats[channel_name]['mean'] = np.mean(channel, axis=1)
            self.data_stats[channel_name]['std'] = np.std(channel, axis=1)
            self.data_stats[channel_name]['RMS'] = np.sqrt(np.mean(x_squared, axis=1))
            self.data_stats[channel_name]['kurtosis'] = stats.kurtosis(channel, axis=1)
            self.data_stats[channel_name]['skewness'] = stats.skew(channel, axis=1)
            self.data_stats[channel_name]['p2p'] = np.ptp(channel, axis=1)  # Shortcut for np.max(x) - np.min(x)
            #timeseries
            
            self.data_stats[channel_name]['crest_factor'] = np.divide(x_max, self.data_stats[channel_name]['RMS'])
            self.data_stats[channel_name]['shape_factor'] = np.divide(self.data_stats[channel_name]['RMS'], x_mean_abs)
            self.data_stats[channel_name]['impulse_factor'] = np.divide(x_max, x_mean_abs)
            self.data_stats[channel_name]['margin_factor'] = np.divide(x_max, np.square(x_mean_abs))
            self.data_stats[channel_name]['energy'] = np.mean(x_squared, axis=1)

            print(f"Channel {channel_name} features extracted") if self.verbose else 0
            #also spectral kurtosis using STFT - tough to tune
            #could get direction and angle like in Sandvik for vibration? better for force really.

            #need to then normalise (x - mean)/std    - gives mean zero and std 1
        self.remove_data()      #remove raw data now features have been extracted
        pass 


    def plot_channel_feature(self, figure_num, rows, cols, idx, channel_name, feature_name):
        data = self.data_stats[channel_name][feature_name]
        N = len(data)

        plt.figure(figure_num)
        plt.subplot(rows, cols, idx)
        plt.plot(data)
        plt.xlabel('Run')
        plt.ylabel(feature_name)
        #plt.title(channel_name)
        plt.grid('show')
        pass


    def plot_all_channel_features(self, figure_num, channel_name):
        c = 1
        for feature in FEATURE_KEYS:
            self.plot_channel_feature(figure_num, 3, 4, c, channel_name, feature)
            c +=1
        plt.suptitle(channel_name)
        pass


    def plot_all_features(self, start_figure_num):
        c = start_figure_num
        for channel_name in CHANNEL_KEYS:
            self.plot_all_channel_features(c, channel_name)
        pass


if __name__ == "__main__":
    analysis = DataAnalysis('./data/Segmented_Linear_Baseline.mat')
    analysis.extract_features()
    
    analysis.plot_all_features(1)
    plt.show()