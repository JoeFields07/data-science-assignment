# Created 08/07/2026
# Author: Joseph Fields
# Description: [[TODO]]

from pathlib import Path
import h5py
import pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

CHANNEL_KEYS = ["PlateLFAccX", "PlateLFAccY", "PlateLFAccZ", "PlateHFAccZ", "SpindleAccX", "SpindleAccY", "SpindleAccZ", "Power"]
FEATURE_KEYS = ['mean', 'std', 'RMS', 'kurtosis', 'skewness', 'p2p', 'crest_factor', 'shape_factor', 'impulse_factor', 'margin_factor', 'energy']
FEATURE_FOLDER = Path("./data_features/")
class FileHandler():
    def __init__(self, filepath, verbose=True):

        self.verbose = verbose
        self.data_filepath = filepath
        FEATURE_FOLDER.mkdir(parents=True, exist_ok=True)       #make feature folder if it doesn't already exist
        
        filename = Path(filepath).stem                          #extract name of data file
        self.feature_filepath = Path(FEATURE_FOLDER) / (filename + ".pkl")

        self.data = {}                                          #initalise data dictionaries
        self.data_stats = {}

        if self.feature_filepath.is_file():                     #if feature data exists, load and don't calculate again
            print("Feature file already exists") if self.verbose else 0
            self.load_feature_file()
        else:
            print("Feature file does not exist") if self.verbose else 0
            self.data = self.load_data_file()
            self.extract_features()
        pass


    def dereference_data(self, file_handle, dataset):
        """Safely reads data, resolving HDF5 references uniformly as lists if they contain arrays."""

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
            return raw_data.flatten()       #I dont think this ever happens

    
    def load_data_file(self):
        """Loads the .mat file and returns a dictionary of the channels"""

        print(f"Loading sensor data from: {self.data_filepath}") if self.verbose else 0
        with h5py.File(self.data_filepath, 'r') as f:
            clean_data = {}
            
            all_keys = list(f.keys())     
            data_group = f[all_keys[1]]   #second key contains the data
            
            for channel_key in data_group.keys():     #could replace with CHANNEL_KEYS
                channel_data = data_group[channel_key]
                
                if isinstance(channel_data, h5py.Dataset):  #could remove
                    clean_data[channel_key] = self.dereference_data(f, channel_data)
        print("Data imported successfully") if self.verbose else 0
        return clean_data


    def load_feature_file(self):
        """Loads the .pkl file containing cached features"""

        print(f"Loading features from: {self.feature_filepath}") if self.verbose else 0
        with open(self.feature_filepath, "rb") as f:
            self.data_stats = pickle.load(f)
        return 
    

    def export_features(self):
        """Exports the features to a .pkl file cache"""

        print(f"Exporting features to: {self.feature_filepath}") if self.verbose else 0
        with open(self.feature_filepath, "wb") as f:
            pickle.dump(self.data_stats, f)
        pass


    def remove_data(self):
        """Removes the raw data to save memory once features have been calculated"""

        self.data = {}
        print("Removed raw data") if self.verbose else 0
        pass

    
    def extract_features(self):
        """
        Extracts features from each channel, creating a two layer dictionary for each channel and feature
        The following features are calculated:
        mean, standard deviation, root mean squared (RMS), kurtosis, skewness, peak-to-peak, crest factor, 
        shape factor, impulse factor, margin factor and energy,
        """

        print("Starting feature extraction") if self.verbose else 0
        for channel_name in CHANNEL_KEYS:
            channel = self.data[channel_name]       #get data for the current channel
            self.data_stats[channel_name] = {}      #need to initialise a dict for each channel
            
            x_squared = np.square(channel)          #pre-calculate components to save time
            x_max = np.max(channel, axis=1)
            x_mean_abs = np.mean(np.abs(channel), axis=1)

            self.data_stats[channel_name]['mean'] = np.mean(channel, axis=1)
            self.data_stats[channel_name]['std'] = np.std(channel, axis=1)
            self.data_stats[channel_name]['RMS'] = np.sqrt(np.mean(x_squared, axis=1))
            self.data_stats[channel_name]['kurtosis'] = stats.kurtosis(channel, axis=1)
            self.data_stats[channel_name]['skewness'] = stats.skew(channel, axis=1)
            self.data_stats[channel_name]['p2p'] = np.ptp(channel, axis=1)
            #timeseries
            
            self.data_stats[channel_name]['crest_factor'] = np.divide(x_max, self.data_stats[channel_name]['RMS'])
            self.data_stats[channel_name]['shape_factor'] = np.divide(self.data_stats[channel_name]['RMS'], x_mean_abs)
            self.data_stats[channel_name]['impulse_factor'] = np.divide(x_max, x_mean_abs)
            self.data_stats[channel_name]['margin_factor'] = np.divide(x_max, np.square(x_mean_abs))
            self.data_stats[channel_name]['energy'] = np.mean(x_squared, axis=1)

            print(f"Channel {channel_name} features extracted") if self.verbose else 0
            #also spectral kurtosis using STFT - tough to tune

        self.remove_data()          #remove raw data now features have been extracted
        self.export_features()      #save newly extracted features
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
            c +=1
        pass


if __name__ == "__main__":
    #file = FileHandler('./data/Segmented_Linear_Baseline.mat')
    file = FileHandler('./data/Segmented_Linear_Heavy.mat')

    #file.plot_all_features(1)
    #plt.show()