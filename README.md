# (MAC4112) data-science-assignment 
This software provides an automated pipeline to import raw AMRC sensor data (from .mat / HDF5 files), handle outlier rejection, extract time-domain statistical features, perform dimensionality reduction via PCA, and train/evaluate Machine Learning classifiers (Random Forest or SVM).

**Dataset:** [Sensor signals for machine tool and process health assessment](https://orda.shef.ac.uk/articles/dataset/Sensor_signals_for_machine_tool_and_process_health_assessment_/24125715)  
**Citation:** _Dominguez Caballero, Javier Alejandro; Moore, James; Stammers, Jon (2023). Sensor signals for machine tool and process health assessment. The University of Sheffield. Dataset._


# Contact
**Author:** Joseph Fields  
**Contact:** jfields1@sheffield.ac.uk


# Contents
```main.py```: The command-line interface (CLI) entry point. It coordinates the execution flow between data_file_helper.py and analysis_helper.py, parsing file arguments and validating that dataset types match appropriately before running the pipeline.

```data_file_helper.py```: Manages file I/O operations. It loads raw structured sensor data from .mat files using HDF5, extracts 11 different statistical features per channel, caches them locally as .pkl files to save processing time on subsequent runs. Also includes plotting functions for data inspection.

```analysis_helper.py```: Contains the analytical and machine learning logic. It performs Median Absolute Deviation (MAD) outlier removal, handles missing data (NaN values), standardizes features, fits/applies Principal Component Analysis (PCA), and handles classifier training and confusion matrix generation. 

```test_data_file_helper.py```: Unit tests for data_file_helper.py functions.

```test_analysis_helper.py```: Unit tests for analysis_helper.py functions.


# Dependancies
Uses Python 3.13.12  
```pip install -r requirements.txt```


# Installation Instructions
```git clone https://github.com/JoeFields07/data-science-assignment```  

The .mat files from the [dataset](https://orda.shef.ac.uk/articles/dataset/Sensor_signals_for_machine_tool_and_process_health_assessment_/24125715) must be downloaded, 
unzipped and placed in a target directory.


# Usage Instructions
The program can be run through the terminal:  
```python main.py -v -d "folder_path" -f "file_name1" "file_name2" "file_name3" ...```

The program can also be run using a .bat file (Windows):  
```launch.bat```  
where the arguments in the .bat file can be modified as needed.  

Unit testing can be performed using:  
```pytest ./```  

## CLI Arguments
- -f, --file (Required): One or more space-separated .mat file names to include in the analysis.

- -d, --directory (Optional): The directory path containing the raw data files. Defaults to ./data/.

- -v, --verbose (Optional): Flag to enable console logging during execution.

**_Note on File Alignment:_** The pipeline validates that you do not mix "Machining" experiment files with other types, as they contain different sensor channel mappings.