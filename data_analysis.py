# Created 08/07/2026
# Author: Joseph Fields
# Description: [[TODO]]

from pathlib import Path
import h5py
import pandas as pd
import numpy as np

class DataAnalysis():
    def __init__(self, filename):
        self.data = self.load_file(filename)
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
            return raw_data.flatten()


    
    def load_file(self, filename):
        # Load the parquet file

        with h5py.File(filename, 'r') as f:
            clean_data = {}
            
            # Find the key that is NOT '#refs#'
            all_keys = list(f.keys()) 
            group_key = all_keys[1] #second key contains the data
            
            # If we found a valid data group, proceed

            struct_group = f[group_key]
            
            for sub_key in struct_group.keys():
                dataset = struct_group[sub_key]
                
                if isinstance(dataset, h5py.Dataset):
                    clean_data[sub_key] = self.dereference_data(f, dataset)
        
        return clean_data


    def extract_features(self):
        #mean, standard deviation, root mean squared (RMS), kurtosis, skewness and peak-to-peak; as well as those less commonly used: crest factor, shape factor, impulse factor, margin factor and energy, 
        return 

if __name__ == "__main__":
    analysis = DataAnalysis('./data/Segmented_Linear_Baseline.mat')
    
    