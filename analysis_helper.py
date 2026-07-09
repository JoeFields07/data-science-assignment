from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class AnalysisHelper():
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        self.full_keys = None
        self.scaler = StandardScaler()

        self.ml_matrix_X = None
        self.ml_matrix_y = None
        self.X_train, self.X_test = None, None      #train/test split datasets
        self.y_train, self.y_test = None, None


    def import_feature_data(self, data_list, channel_keys, feature_keys): 
        
        self.full_keys = [f"{ch}_{feat}" for ch in channel_keys for feat in feature_keys]   #create names by combining both keys
        column_collector = {name: [] for name in self.full_keys}        #make collector
        labels_list = []

        for dataset_idx, d in enumerate(data_list, start=1):
            # Pick the first available channel and feature to get length
            first_ch = channel_keys[0]
            first_feat = feature_keys[0]
            num_samples = len(d[first_ch][first_feat])  # e.g., 420

            # Generate and store the labels for this dataset
            labels_list.append(np.repeat(dataset_idx, num_samples))
            # --------------------------

            # Gather the actual features
            for ch in channel_keys:
                for feat in feature_keys:
                    key_name = f"{ch}_{feat}"
                    column_collector[key_name].append(d[ch][feat])
                
        # 3. Compile the final X Matrix (Concatenate chunks first, then stack side-by-side)
        self.ml_matrix_X = np.column_stack([np.concatenate(column_collector[k]) for k in self.full_keys])
        self.ml_matrix_y = np.concatenate(labels_list)


    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the ML matrix and labels into training and testing sets.
        Default is an 80/20 split.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.ml_matrix_X, 
        self.ml_matrix_y, 
        test_size=test_size,      # 0.2 means 20% goes to testing, 80% to training
        random_state=random_state, # Acts as a seed so your split is identical every time you run it
        stratify=self.ml_matrix_y)  # Optional: Ensures train/test sets get equal proportions of classes 1, 2, and 3


    def preprocess_data(self):
        """
        Scales the data so mean is zero and std is 1. 
        """
        #may need to add something to split into test and training?
        self.ml_matrix_X = self.scaler.fit_transform(self.ml_matrix_X)

