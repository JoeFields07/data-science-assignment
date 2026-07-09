# Created 08/07/2026
# Author: Joseph Fields
# Description: [[TODO]]

from pathlib import Path
import h5py
import pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

ALL_CHANNEL_KEYS = ["PlateLFAccX", "PlateLFAccY", "PlateLFAccZ", "PlateHFAccZ", "SpindleAccX", "SpindleAccY", "SpindleAccZ", "Power"]
MACHINING_CHANNEL_KEYS = ["PlateLFAccX", "PlateLFAccY", "PlateLFAccZ", "SpindleX", "SpindleY", "SpindleZ", "Power"]
FEATURE_KEYS = ['mean', 'std', 'RMS', 'kurtosis', 'skewness', 'p2p', 'crest_factor', 'shape_factor', 'impulse_factor', 'margin_factor', 'energy']
FEATURE_FOLDER = Path("./data_features/")

class FileHelper():
    def __init__(self, filepath, verbose=True):

        self.verbose = verbose
        self.data_filepath = filepath
        FEATURE_FOLDER.mkdir(parents=True, exist_ok=True)       #make feature folder if it doesn't already exist
        
        filename = Path(filepath).stem                          #extract name of data file
        self.feature_filepath = Path(FEATURE_FOLDER) / (filename + ".pkl")

        self.data = {}                                          #initalise data dictionaries
        self.data_stats = {}

        if "Machining" in filename:                             #the Machining files have different keys
            self.channel_keys = MACHINING_CHANNEL_KEYS
        else:
            self.channel_keys = ALL_CHANNEL_KEYS

        if self.feature_filepath.is_file():                     #if feature data exists, load and don't calculate again
            print("Feature file already exists") if self.verbose else 0
            self.load_feature_file()
        else:
            print("Feature file does not exist") if self.verbose else 0
            self.data = self.load_data_file()
            self.extract_features()


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
            
            for channel_key in self.channel_keys:       #some datasets have different labels
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
    

    def export_features(self):
        """Exports the features to a .pkl file cache"""

        print(f"Exporting features to: {self.feature_filepath}") if self.verbose else 0
        with open(self.feature_filepath, "wb") as f:
            pickle.dump(self.data_stats, f)


    def remove_data(self):
        """Removes the raw data to save memory once features have been calculated"""

        self.data = {}
        print("Removed raw data") if self.verbose else 0

    
    def extract_features(self):
        """
        Extracts features from each channel, creating a two layer dictionary for each channel and feature
        The following features are calculated:
        mean, standard deviation, root mean squared (RMS), kurtosis, skewness, peak-to-peak, crest factor, 
        shape factor, impulse factor, margin factor and energy,
        """

        print("Starting feature extraction") if self.verbose else 0
        for channel_name in self.channel_keys:
            channel = self.data[channel_name]       #get data for the current channel
            self.data_stats[channel_name] = {}      #need to initialise a dict for each channel
            
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
            self.data_stats[channel_name]['mean'] = means
            self.data_stats[channel_name]['std'] = stds
            self.data_stats[channel_name]['RMS'] = rms_vals
            self.data_stats[channel_name]['kurtosis'] = kurtosis
            self.data_stats[channel_name]['skewness'] = skewness
            self.data_stats[channel_name]['p2p'] = p2p
            self.data_stats[channel_name]['energy'] = energy
            self.data_stats[channel_name]['crest_factor']   = np.divide(x_max, rms_vals)
            self.data_stats[channel_name]['shape_factor']   = np.divide(rms_vals, x_mean_abs)
            self.data_stats[channel_name]['impulse_factor'] = np.divide(x_max, x_mean_abs)
            self.data_stats[channel_name]['margin_factor']  = np.divide(x_max, np.square(x_mean_abs))

            print(f"Channel {channel_name} features extracted") if self.verbose else 0
            #also spectral kurtosis using STFT - tough to tune

        self.remove_data()          #remove raw data now features have been extracted
        self.export_features()      #save newly extracted features


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


    def plot_all_channel_features(self, figure_num, channel_name):
        c = 1
        for feature in FEATURE_KEYS:
            self.plot_channel_feature(figure_num, 3, 4, c, channel_name, feature)
            c +=1
        plt.suptitle(channel_name)


    def plot_all_features(self, start_figure_num):
        c = start_figure_num
        for channel_name in self.channel_keys:
            self.plot_all_channel_features(c, channel_name)
            c +=1


if __name__ == "__main__":
    #file = FileHelper('./data/Segmented_Linear_Baseline.mat')
    file = FileHelper('./data/Segmented_Linear_Override.mat')

    #file.plot_all_features(1)
    #plt.show()