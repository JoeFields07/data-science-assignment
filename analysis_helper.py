from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class AnalysisHelper():
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.full_keys = None

        self.ml_matrix_X = None
        self.ml_matrix_y = None
        self.X_train, self.X_test = None, None      #train/test split datasets
        self.y_train, self.y_test = None, None
        self.X_train_PCA, self.X_test_PCA = None, None

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)


    def import_feature_data(self, data_list, channel_keys, feature_keys): 
        """
        Combine multiple dictionaries from the helper to make a ML friendly matrix, also generate classification labels. 
        """
        print(f"Combining {len(data_list)} datasets") if self.verbose else 0
        self.full_keys = [f"{ch}_{feat}" for ch in channel_keys for feat in feature_keys]   #create names by combining both keys
        column_collector = {name: [] for name in self.full_keys}        #make collector
        labels_list = []

        for d_idx, data in enumerate(data_list, start=0):
            # Pick the first available channel and feature to get length
            num_samples = len(data[channel_keys[0]][feature_keys[0]]) 
            # Generate and store the labels for this dataset
            labels_list.append(np.repeat(d_idx, num_samples))
            # Gather the actual features
            for ch in channel_keys:
                for feat in feature_keys:
                    key_name = f"{ch}_{feat}"
                    column_collector[key_name].append(data[ch][feat])
                
        # Compile the final X Matrix (Concatenate chunks first, then stack side-by-side)
        self.ml_matrix_X = np.column_stack([np.concatenate(column_collector[k]) for k in self.full_keys])
        self.ml_matrix_y = np.concatenate(labels_list)

        print(f"X matrix of shape {np.shape(self.ml_matrix_X)} created") if self.verbose else 0


    def remove_outliers(self):
        # 1. Calculate the mean and standard deviation for each column
        mean = np.mean(self.ml_matrix_X, axis=0)
        std = np.std(self.ml_matrix_X, axis=0)
        
        # Avoid division by zero if a feature has zero variance
        std[std == 0] = 1.0

        # 2. Calculate the absolute Z-scores for every cell
        # Formula: |(x - mean) / std|
        z_scores = np.abs((self.ml_matrix_X - mean) / std)

        # 3. Create a boolean mask of rows to KEEP
        keep_mask = np.all(z_scores < 3, axis=1)

        # 4. Filter both X and y simultaneously
        original_shape = self.ml_matrix_X.shape[0]
        self.ml_matrix_X = self.ml_matrix_X[keep_mask]
        self.ml_matrix_y = self.ml_matrix_y[keep_mask]
        
        removed_count = original_shape - self.ml_matrix_X.shape[0]
        print(f"{removed_count} outliers removed") if self.verbose else 0


    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Splits the ML matrix and labels into training and testing sets.
        Default is an 80/20 split. Scales the data so mean is zero and std is one. 
        """
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.ml_matrix_X, 
        self.ml_matrix_y, 
        test_size = test_size,      # 0.2 means 20% goes to testing, 80% to training
        random_state = random_state, # Acts as a seed so your split is identical every time you run it
        stratify = self.ml_matrix_y)  # Optional: Ensures train/test sets get equal proportions of classes 1, 2, and 3
        print(f"Split dataset {1.0-test_size} train and {test_size} test") if self.verbose else 0

        self.X_train = self.scaler.fit_transform(self.X_train)      #train scaler and scale data
        self.X_test = self.scaler.transform(self.X_test)            #only scale data using previous parameters
        print(f"Scaled X_train and X_test matrix") if self.verbose else 0


    def train_PCA(self):
        """
        Obtains the PCA transform using the X_train dataset
        """
        self.X_train_PCA = self.pca.fit_transform(self.X_train)     #train and apply PCA
        print(f"Trained and applied PCA to X_train") if self.verbose else 0
        pass
    

    def apply_PCA(self):
        """
        Applies the PCA transform to the X_test dataset
        """
        self.X_test_PCA = self.pca.transform(self.X_test)           #only apply PCA
        print(f"Applied PCA to X_test using existing parameters") if self.verbose else 0


    def plot_PCA(self, fig_num, legend_labels):
        x = self.X_train_PCA[:, 0]
        y = self.X_train_PCA[:, 1]
        z = self.X_train_PCA[:, 2]
        self.plot_data(fig_num, x, y, z, "PCA1", "PCA2", "PCA3", legend_labels)


    def plot_feature(self, fig_num, name_x, name_y, name_z, legend_labels):
        x_idx = self.full_keys.index(name_x)
        y_idx = self.full_keys.index(name_y)
        z_idx = self.full_keys.index(name_z)
        x = self.X_train[:, x_idx]
        y = self.X_train[:, y_idx]
        z = self.X_train[:, z_idx]
        self.plot_data(fig_num, x, y, z, name_x, name_y, name_z, legend_labels)


    def plot_data(self, fig_num, x, y, z, x_label, y_label, z_label, legend_labels):
        # Set up the 3D plotting environment
        fig = plt.figure(fig_num, figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Create the scatter plot using 'y_train' for the color mapping (c)
        # 'cmap' defines the color palette, 'edgecolor' makes points crisp
        scatter = ax.scatter(
            x, y, z, 
            c=self.y_train, 
            cmap='viridis', 
            s=30, 
            edgecolor='k', 
            alpha=0.8
        )

        # Add labels, title, and a clean color legend
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        #ax.set_title('')

        # Create a nice legend showing which color represents which dataset label (1, 2, or 3)
        handles, _ = scatter.legend_elements()
        ax.legend(handles=handles,
                labels=legend_labels, 
                title="Datasets")
        
        plt.show(block = False)

