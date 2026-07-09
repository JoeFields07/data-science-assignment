from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FileHandler():
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.full_dataset = {}
        self.full_keys = []
        pass

    def import_feature_data(self, channel_keys, feature_keys, labels, *data_list):  #data is combined into one interator
        if len(labels) == len(data_list):       #check number of labels matches number of datasets
            full_feature_names = [f"{ch}_{feat}" for ch in channel_keys for feat in feature_keys]
            
            for data in data_list:
                for channel_name in channel_keys:
                    for feature_name in feature_keys:
                        feature = data[channel_name][feature_name]
                        new_key = channel_name + "_" + feature_name
                        
                        if new_key not in self.full_keys:   # rack unique keys
                            self.full_keys.append(new_key)

                        if new_key not in self.full_dataset:
                            self.full_dataset[new_key] = [] #Initialize the list in dict if new key

                        self.full_dataset[new_key].append(feature)

        else:
            print("Number of labels does not match datasets")