# Created 08/07/2026
# Author: Joseph Fields
# Description: Manages file I/O operations. It loads raw structured sensor data 
# from .mat files using HDF5, extracts 11 different statistical features per channel, 
# caches them locally as .pkl files to save processing time on subsequent runs. 
# Also includes plotting functions for data inspection.

from pathlib import Path
import h5py
import pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

ALL_CHANNEL_KEYS = ["PlateLFAccX", "PlateLFAccY", "PlateLFAccZ", "PlateHFAccZ", "SpindleAccX", "SpindleAccY", "SpindleAccZ", "Power"]
MACH_CHANNEL_KEYS = ["PlateLFAccX", "PlateLFAccY", "PlateLFAccZ", "SpindleLoad", "SpindleX", "SpindleY", "SpindleZ", "Power"]
ALL_PLOT_KEYS = ["SpindleAccX_p2p", "SpindleAccY_kurtosis", "SpindleAccX_mean"]
MACH_PLOT_KEYS = ["SpindleX_p2p", "SpindleY_kurtosis", "SpindleX_mean"]
FEATURE_KEYS = ['mean', 'std', 'RMS', 'kurtosis', 'skewness', 'p2p', 'crest_factor', 'shape_factor', 'impulse_factor', 'margin_factor', 'energy']
FEATURE_FOLDER = Path("./data_features/")

class FileHelper():
    def __init__(self, filepath, verbose=False):

        self.verbose = verbose
        self.data_filepath = Path(filepath)
        FEATURE_FOLDER.mkdir(parents=True, exist_ok=True)       #make feature folder if it doesn't already exist
        
        filename = Path(filepath).stem                          #extract name of data file
        split_filename = filename.split('_')
        self.experiment = split_filename[1]
        self.variant = split_filename[2]

        self.feature_filepath = Path(FEATURE_FOLDER) / (filename + ".pkl")
        
        self.data = {}
        self.data_stats = {}

        if self.experiment == "Machining":                 #the Machining files have different keys
            self.channel_keys = MACH_CHANNEL_KEYS
            self.plot_keys = MACH_PLOT_KEYS
        else:
            self.channel_keys = ALL_CHANNEL_KEYS
            self.plot_keys = ALL_PLOT_KEYS



    def dereference_data(self, file_handle, dataset):
        """
        Safely reads data, resolving HDF5 references as lists if they contain arrays.
        """
        raw_data = np.array(dataset).T
        
        if raw_data.dtype == object and len(raw_data) > 0:      #check data is valid
            resolved_list = []
            for ref in raw_data.flatten():
                if isinstance(ref, h5py.Reference):
                    ref_dataset = file_handle[ref]
                    actual_val = np.array(ref_dataset).T.flatten()
                    resolved_list.append(actual_val.tolist())   #ensure every row is a list

                else:
                    # If it's a null/empty reference, provide an empty list to keep types uniform
                    resolved_list.append([])

            return resolved_list
        
        else:
            # For standard purely numeric/flat columns, return as a flat numpy array
            return raw_data.flatten()       #Don't think this ever happens

    
    def load_data_file(self):
        """
        Loads the .mat file and returns a dictionary of the channels.
        """
        print(f"Loading sensor data from: {self.data_filepath}") if self.verbose else 0
        with h5py.File(self.data_filepath, 'r') as f:
            clean_data = {}
            all_keys = list(f.keys())     
            data_group = f[all_keys[1]]   #second key contains the data
            
            for channel_key in self.channel_keys:       #some datasets have different labels
                channel_data = data_group[channel_key]
                
                if isinstance(channel_data, h5py.Dataset):  #could remove
                    clean_data[channel_key] = self.dereference_data(f, channel_data)
        
        print("Data imported successfully") if self.verbose else 0
        
        return clean_data


    def load_feature_file(self):
        """
        Returns the cached features inside the .pkl file.
        """
        print(f"Loading features from: {self.feature_filepath}") if self.verbose else 0
        with open(self.feature_filepath, "rb") as f:
            return pickle.load(f)
    

    def export_features(self):
        """
        Exports the features to a .pkl file cache.
        """
        print(f"Exporting features to: {self.feature_filepath}") if self.verbose else 0
        with open(self.feature_filepath, "wb") as f:
            pickle.dump(self.data_stats, f)


    def remove_data(self):
        """
        Removes the raw data to save memory once features have been calculated.
        """
        self.data = {}
        print("Removed raw data") if self.verbose else 0

    
    def extract_features(self):
        """
        Extracts features from each channel, creating a two layer dictionary for each channel and feature
        The following features are calculated:
        mean, standard deviation, root mean squared (RMS), kurtosis, skewness, peak-to-peak, crest factor, 
        shape factor, impulse factor, margin factor and energy.
        """
        # ~3 mins per .mat file. Efficiency could be improved with np arrays instead of dictionaries 
        # However only needs to be ran once and dicts are very human readable. 
        data_stats = {}
        print("Starting feature extraction") if self.verbose else 0
        
        for channel_name in self.channel_keys:
            channel = self.data[channel_name]       #get data for the current channel
            data_stats[channel_name] = {}      #need to initialise a dict for each channel
            
            #Process metrics row-by-row using list comprehensions to handle unequal lengths
            means       = np.array([np.mean(row) for row in channel])
            stds        = np.array([np.std(row) for row in channel])
            x_max       = np.array([np.max(row) for row in channel])
            x_mean_abs  = np.array([np.mean(np.abs(row)) for row in channel])

            #Pre-calculate row-by-row components
            rms_vals    = np.array([np.sqrt(np.mean(np.square(row))) for row in channel])
            kurtosis    = np.array([stats.kurtosis(row) for row in channel])
            skewness    = np.array([stats.skew(row) for row in channel])
            p2p         = np.array([np.ptp(row) for row in channel])
            energy      = np.array([np.mean(np.square(row)) for row in channel])

            #Store the calculated 1D arrays into dictionary
            data_stats[channel_name]['mean'] = means
            data_stats[channel_name]['std'] = stds
            data_stats[channel_name]['RMS'] = rms_vals
            data_stats[channel_name]['kurtosis'] = np.nan_to_num(kurtosis, nan=0.0)        #sometimes these can be NaN due to python precision loss
            data_stats[channel_name]['skewness'] = np.nan_to_num(skewness, nan=0.0)
            data_stats[channel_name]['p2p'] = p2p
            data_stats[channel_name]['energy'] = energy
            data_stats[channel_name]['crest_factor']   = np.divide(x_max, rms_vals)
            data_stats[channel_name]['shape_factor']   = np.divide(rms_vals, x_mean_abs)
            data_stats[channel_name]['impulse_factor'] = np.divide(x_max, x_mean_abs)
            data_stats[channel_name]['margin_factor']  = np.divide(x_max, np.square(x_mean_abs))

            print(f"Channel {channel_name} features extracted") if self.verbose else 0

        return data_stats
    

    def plot_channel_feature(self, figure_num, rows, cols, idx, channel_name, feature_name):
        """
        Plot a specific feature in a channel for visualisation.
        """
        data = self.data_stats[channel_name][feature_name]
        N = len(data)

        plt.figure(figure_num)
        plt.subplot(rows, cols, idx)
        plt.plot(data)
        plt.xlabel('Run')
        plt.ylabel(feature_name)
        #plt.title(channel_name)
        plt.grid('show')


    def plot_all_channel_features(self, figure_num, channel_name):
        """
        Plot a all features in a specific channel for visualisation.
        """
        c = 1
        for feature in FEATURE_KEYS:
            self.plot_channel_feature(figure_num, 3, 4, c, channel_name, feature)
            c +=1
        plt.suptitle(channel_name)


    def plot_all_features(self, start_figure_num):
        """
        Plot all features in all channels for visualisation. (Memory intensive)
        """
        c = start_figure_num
        for channel_name in self.channel_keys:
            self.plot_all_channel_features(c, channel_name)
            c +=1



if __name__ == "__main__":
    file = FileHelper('./data/Segmented_Linear_Baseline.mat', verbose=True)     #for debugging
    file.plot_all_features(1)
    plt.show()