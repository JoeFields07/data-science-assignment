# (MAC4112) data-science-assignment 
This software provides an automated pipeline to import raw AMRC sensor data (from .mat / HDF5 files), handle outlier rejection, extract time-domain statistical features, perform dimensionality reduction via PCA, and train/evaluate Machine Learning classifiers (Random Forest or SVM).

**Dataset:** [Sensor signals for machine tool and process health assessment](https://orda.shef.ac.uk/articles/dataset/Sensor_signals_for_machine_tool_and_process_health_assessment_/24125715)  
**Citation:** _Dominguez Caballero, Javier Alejandro; Moore, James; Stammers, Jon (2023). Sensor signals for machine tool and process health assessment. The University of Sheffield. Dataset._


# Contact
**Author:** Joseph Fields  
**Contact:** jfields1@sheffield.ac.uk


# Contents
`main.py`: The command-line interface (CLI) entry point. It coordinates the execution flow between data_file_helper.py and analysis_helper.py, parsing file arguments and validating that dataset types match appropriately before running the pipeline.

`data_file_helper.py`: Manages file I/O operations. It loads raw structured sensor data from .mat files using HDF5, extracts 11 different statistical features per channel, caches them locally as .pkl files to save processing time on subsequent runs. Also includes plotting functions for data inspection.

`analysis_helper.py`: Contains the analytical and machine learning logic. It performs Median Absolute Deviation (MAD) outlier removal, handles missing data (NaN values), standardizes features, fits/applies Principal Component Analysis (PCA), and handles classifier training and confusion matrix generation. 

`test_data_file_helper.py`: Unit tests for data_file_helper.py functions.

`test_analysis_helper.py`: Unit tests for analysis_helper.py functions.


# Dependancies
Uses Python 3.13.12  
`pip install -r requirements.txt`


# Installation Instructions
`git clone https://github.com/JoeFields07/data-science-assignment`  

The .mat files from the [dataset](https://orda.shef.ac.uk/articles/dataset/Sensor_signals_for_machine_tool_and_process_health_assessment_/24125715) must be downloaded, 
unzipped and placed in a target directory.


# Usage Instructions
The program can be run through the terminal:  
`python main.py -v -d "folder_path" -f "file_name1" "file_name2" "file_name3" ...`

The program can also be run using a .bat file (Windows) where the arguments can be modified as needed:  
`launch.bat`  


Further customisation can be performed by modifying some of the constants in main.py.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `MAD_FILTER_THRESHOLD` | 10.0 | Threshold for Median Absolute Difference (MAD) filtering. |
| `PCA_COMPONENTS` | 10 | Number of PCA components to calculate. More PCA components means the classifier has more information to use during classification and will improve accuracy. |
| `CLASSIFIER` | 'RF' | Select the classifier - 'SVM' for SVM or 'RF' for Random Forest. |
| `RF_N_ESTIMATORS` | 100 | Random Forest classifier 'n_estimators' parameter. |
| `SVM_C` | 1.0 | SVM Classifier 'C' parameter. |
| `RANDOM_SEED` | 42 | Random seed used for train_test_split, PCA, and classifier training. |
| `TRAIN_TEST_SPLIT` | 0.2 | Proportion of total set to the test set. 0.2 means 80% train and 20% test. |


Unit testing can also be performed using:  
`pytest ./`  

## CLI Arguments
| Argument | Required | Description |
| :--- | :--- | :--- |
| -f, --file | Y | One or more space-separated .mat file names to include in the analysis. |
| -d, --directory | N | The directory path containing the raw data files. Defaults to ./data/. |
| -v, --verbose | N | Flag to enable console logging during execution. |

**_Note on File Alignment:_** The pipeline validates that you do not mix "Machining" experiment files with other types, as they contain different sensor channel mappings.