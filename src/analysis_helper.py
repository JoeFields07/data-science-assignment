# Created 10/07/2026
# Author: Joseph Fields
# Description: Contains the analytical and machine learning logic. It performs 
# Median Absolute Deviation (MAD) outlier removal, handles missing data 
# (NaN values), standardizes features, fits/applies Principal Component Analysis 
# (PCA), and handles classifier training and confusion matrix generation.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


class AnalysisHelper():
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.full_keys = None

        self.ml_matrix_X = None
        self.ml_matrix_y = None
        self.X_train, self.X_test = None, None          #train/test split datasets
        self.y_train, self.y_test = None, None
        self.X_train_PCA, self.X_test_PCA = None, None  #PCA datasets
        self.y_pred = None

        self.scaler = StandardScaler()
        self.pca = None                                 #can increase to improve ML prediction 
        self.clf = None


    def remove_outliers_aligned(self, X_chunk, y_chunk, threshold_limit):
        """
        Removes sensor spikes using Median Absolute Deviation (MAD).
        Safely ignores and handles NaN values to prevent whole-matrix erasure.
        """
        # Replace any NaNs with the column median so they don't break the math
        if np.any(np.isnan(X_chunk)):
            # Calculate medians ignoring existing NaNs
            col_medians = np.nanmedian(X_chunk, axis=0)
            # If an entire column was NaN, fallback to 0.0
            col_medians[np.isnan(col_medians)] = 0.0

            # Find where the NaNs are and fill them
            inds = np.where(np.isnan(X_chunk))
            X_chunk[inds] = np.take(col_medians, inds[1])

        # Calculate robust median and MAD along axis 0
        median = np.median(X_chunk, axis=0)
        abs_deviation = np.abs(X_chunk - median)
        mad = np.median(abs_deviation, axis=0)

        # Handle flatlines (replace near-zeros with 1.0)
        mad[mad < 1e-6] = 1.0

        # Broadcast across the 2D matrix
        mad_2d = mad[np.newaxis, :]  
        robust_z_scores = 0.6745 * abs_deviation / mad_2d

        # If any cell is still NaN (highly unlikely now), treat it as False to be safe
        keep_mask = np.all(np.nan_to_num(robust_z_scores, nan=np.inf) < threshold_limit, axis=1)

        # Apply mask
        filtered_X = X_chunk[keep_mask]
        filtered_y = y_chunk[keep_mask]
        removed_count = X_chunk.shape[0] - filtered_X.shape[0]

        return filtered_X, filtered_y, removed_count


    def import_feature_data(self, data_list, channel_keys, feature_keys, threshold_limit=4.0): 
        """
        Combine multiple dictionaries from the helper into an ML matrix,
        filtering outliers per feature while preserving strict row alignment.
        """
        print(f"Combining {len(data_list)} datasets") if self.verbose else 0
        self.full_keys = [f"{ch}_{feat}" for ch in channel_keys for feat in feature_keys]   

        clean_X_chunks = []
        clean_y_chunks = []
        total_removed = 0

        for d_idx, data in enumerate(data_list, start=0):
            # Build a temporary matrix to keep rows aligned
            dataset_columns = []
            for ch in channel_keys:
                for feat in feature_keys:
                    dataset_columns.append(data[ch][feat])

            X_chunk = np.column_stack(dataset_columns)
            y_chunk = np.repeat(d_idx, X_chunk.shape[0])

            # Pass the aligned chunk to the filtering function
            filtered_X, filtered_y, removed = self.remove_outliers_aligned(X_chunk, y_chunk, threshold_limit)
            #filtered_X, filtered_y, removed = X_chunk, y_chunk, std_limit
            print("Error: entire dataset filtered out, increase threshold") if filtered_X.shape[0] == 0 else 0
            total_removed += removed

            # Collect the clean chunks
            clean_X_chunks.append(filtered_X)
            clean_y_chunks.append(filtered_y)

        print(f"Total aligned rows removed as outliers: {total_removed}") if self.verbose else 0
        # Combine all clean dataset chunks into final ML matrices
        self.ml_matrix_X = np.concatenate(clean_X_chunks, axis=0)
        self.ml_matrix_y = np.concatenate(clean_y_chunks, axis=0)

        print(f"Final X matrix has {np.shape(self.ml_matrix_X)[0]} rows and {np.shape(self.ml_matrix_X)[1]} feature columns") if self.verbose else 0


    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Splits the ML matrix and labels into training and testing sets.
        Default is an 80/20 split. Scales the data so mean is zero and std is one. 
        """
        unique, counts = np.unique(self.ml_matrix_y, return_counts=True)
        print("Class counts in full dataset:", dict(zip(unique.tolist(), counts.tolist())))
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.ml_matrix_X, 
            self.ml_matrix_y, 
            test_size = test_size,
            random_state = random_state,    # Random seed for repeatability
            stratify = self.ml_matrix_y)    # Ensures train/test sets get equal proportions of classes 1, 2, and 3
        
        print(f"Split dataset {1.0-test_size} train and {test_size} test") if self.verbose else 0

        self.X_train = self.scaler.fit_transform(self.X_train)      #train scaler and scale data
        self.X_test = self.scaler.transform(self.X_test)            #only scale data using previous parameters
        print(f"Scaled X_train and X_test matrix features") if self.verbose else 0


    def train_PCA(self, n_components = 3, random_state=42):
        """
        Obtains the PCA transform using the X_train dataset.
        """
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.X_train_PCA = self.pca.fit_transform(self.X_train)     #train and apply PCA
        print(f"Trained and applied PCA to X_train") if self.verbose else 0
    

    def apply_PCA(self):
        """
        Applies the PCA transform to the X_test dataset.
        """
        self.X_test_PCA = self.pca.transform(self.X_test)           #only apply PCA
        print(f"Applied PCA to X_test using trained PCA transform") if self.verbose else 0


    def train_classifier(self, classifier='SVM', C=1.0, n_estimators=100, random_state=42):
        """
        Train a SVM or Random Forest classifier.
        """
        if classifier == 'SVM':
            self.clf = SVC(kernel='rbf', C=C, random_state=random_state)
            print(f"Trained SVM classifier") if self.verbose else 0

        elif classifier == 'RF':
            self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            print(f"Trained Random Forest classifier") if self.verbose else 0

        else:
            print("Error: invalid classifier name")
        
        self.clf.fit(self.X_train_PCA, self.y_train)        #train selected classifier


    def predict_classifier(self):
        """
        Predict test data using classifier, plot confusion matrix.
        """
        self.y_pred = self.clf.predict(self.X_test_PCA)     #predict using trained classifier
        print(f"Prediction result obtained using trained classifier") if self.verbose else 0
    

    def plot_PCA(self, fig_num, legend_labels):
        """
        Get PCA 1 2 and 3 and plot them on a 3D scatter plot.
        """
        x = self.X_train_PCA[:, 0]
        y = self.X_train_PCA[:, 1]
        z = self.X_train_PCA[:, 2]
        title = "PCA Results"
        self.plot_data(fig_num, x, y, z, "PCA1", "PCA2", "PCA3", title, legend_labels)


    def plot_feature(self, fig_num, plot_keys, legend_labels):
        """
        Plot the 3 specified features.
        """
        if len(plot_keys) == 3:
            x_idx = self.full_keys.index(plot_keys[0])
            y_idx = self.full_keys.index(plot_keys[1])
            z_idx = self.full_keys.index(plot_keys[2])
            x = self.X_train[:, x_idx]
            y = self.X_train[:, y_idx]
            z = self.X_train[:, z_idx]
            title = "Selected Feature Results"
            self.plot_data(fig_num, x, y, z, plot_keys[0], plot_keys[1], plot_keys[2], title, legend_labels)

        else:
            print("Error: Invalid number of plotting features")


    def plot_data(self, fig_num, x, y, z, x_label, y_label, z_label, title, legend_labels):
        """
        Standard 3D plotting function.
        """
        fig = plt.figure(fig_num, figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Create the scatter plot using 'y_train' for the color mapping (c)
        scatter = ax.scatter(
            x, y, z, 
            c=self.y_train, 
            cmap='plasma', 
            s=30, 
            edgecolor='k', 
            alpha=0.8)

        # Add labels and a clean color legend
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.set_title(title)

        # Create a nice legend showing which color represents which dataset label (1, 2, or 3)
        handles, _ = scatter.legend_elements()
        ax.legend(handles=handles,
                labels=legend_labels, 
                title="Datasets")
        
        plt.show(block = False)


    def plot_classifier(self, fig_num, classifier, labels):
        """
        Display confusion matrix of classification result.
        """
        print("#### " + classifier + " ####")
        print(classification_report(self.y_test, self.y_pred, target_names=labels))
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        # Use num=figure_num to set the figure ID explicitly
        fig, ax = plt.subplots(num=fig_num, figsize=(10, 8)) 
        
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        plt.title(classifier + " Confusion Matrix")
        ax.set_xticklabels(labels, rotation=30, ha='right') #rotate labels by Xdeg to fit
        plt.tight_layout()
        plt.show(block = False)
