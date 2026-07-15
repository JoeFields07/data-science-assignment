# Created 08/07/2026
# Author: Joseph Fields
# Description: The command-line interface (CLI) entry point. It coordinates the execution 
# flow between data_file_helper.py and analysis_helper.py, parsing file arguments and 
# validating that dataset types match appropriately before running the pipeline.

from src.data_file_helper import FileHelper, ALL_CHANNEL_KEYS, MACH_CHANNEL_KEYS, FEATURE_KEYS
from src.analysis_helper import AnalysisHelper
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent      #get current folder (project root)
MAD_FILTER_THRESHOLD = 25.0     # threshold for MAD filtering (default 10.0)
PCA_COMPONENTS = 10             # number of PCA components to calculate (default 10)
CLASSIFIER = 'RF'               # classifier ('SVM' or 'RF')
RF_N_ESTIMATORS = 100           # Random Forest classifier 'n_estimators' parameter (default 100)
SVM_C = 1.0                     # SVM classifier 'C' parameter (default 1.0)
RANDOM_SEED = 42                # random seed used
TRAIN_TEST_SPLIT = 0.2          # proportion of the test set (train is 1-test)


def main():
    parser = argparse.ArgumentParser(description="Process data file/s.")
    
    parser.add_argument(
        "-d", "--directory",
        type=str,
        required=False,
        default="data",
        help="Folder containing cut files (default is './data/')"
    )

    parser.add_argument(                            #returns none if nothing entered
        "-f", "--file", 
        type=str,
        nargs="+",
        required=True, 
        help="Enter the path/s of the .mat file for analysis"
    )

    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Parse the arguments from the terminal
    args = parser.parse_args() 
    
    data_obj_list = []
    experiment_list = []
    c = 0

    for filename in args.file:                      #create list of FileHelper instances for each .mat file
        data_obj_list.append(FileHelper(project_root = PROJECT_ROOT,
                                        filepath = Path(args.directory + "/" + filename), 
                                        verbose = args.verbose))
        experiment_list.append(data_obj_list[c].experiment)
        c+=1

    #Valid if either "Machining" isn't in list OR if "Machining" is only thing in list
    valid_file_combination = ("Machining" not in experiment_list) or (set(experiment_list) == {"Machining"})

    if valid_file_combination:
        stats_list = []
        labels = []
        for obj in data_obj_list:                   #loop through list of FileHelper instances
            if obj.feature_file_path.is_file():      #if feature data cached, load and don't calculate again
                print("Feature file already exists") if args.verbose else 0
                obj.data_stats = obj.load_feature_file()

            else:
                if obj.data_file_path.is_file():     #if feature file doesn't exist, calculate it
                    print("Feature file does not exist") if args.verbose else 0
                    obj.data = obj.load_data_file()
                    obj.data_stats = obj.extract_features()
                    obj.export_features()           #save newly extracted features
                    obj.remove_data()               #remove raw data after features extraction
        
                else:
                    FileNotFoundError("Error: Invalid data file name")
                    
            stats_list.append(obj.data_stats)
            labels.append(obj.experiment + obj.variant)
        
        channel_keys = data_obj_list[0].channel_keys    #channel_keys should be same for all files
        plot_keys = data_obj_list[0].plot_keys

        analysis = AnalysisHelper(verbose=args.verbose) 

        #combine FileHelper dicts into ML friendly matrix
        analysis.import_feature_data(data_list = stats_list, 
                                     channel_keys = channel_keys, 
                                     feature_keys = FEATURE_KEYS, 
                                     threshold_limit = MAD_FILTER_THRESHOLD)   
        
        analysis.preprocess_data(test_size = TRAIN_TEST_SPLIT, 
                                 random_state = RANDOM_SEED)  #split and standardise
        
        analysis.train_PCA(n_components = PCA_COMPONENTS, 
                           random_state = RANDOM_SEED)    #train PCA on X_train
        analysis.apply_PCA()                            #apply trained PCA to X_test
        
        analysis.plot_PCA(fig_num = 1, legend_labels = labels)
        analysis.plot_feature(fig_num = 2, plot_keys = plot_keys, legend_labels = labels)

        analysis.train_classifier(classifier = 'RF', 
                                  C = SVM_C, 
                                  n_estimators = RF_N_ESTIMATORS, 
                                  random_state = RANDOM_SEED)
        analysis.predict_classifier()                   #train and predict with classifier
        analysis.plot_classifier(fig_num = 3, classifier = 'RF', labels = labels)
    
        analysis.train_classifier(classifier = 'SVM', 
                                  C = SVM_C, 
                                  n_estimators = RF_N_ESTIMATORS, 
                                  random_state = RANDOM_SEED)
        analysis.predict_classifier()                   #train and predict with classifier
        analysis.plot_classifier(fig_num = 4, classifier = 'SVM', labels = labels)

    else:
        print("Error: Invalid combination of data files")



if __name__ =="__main__":
    main()
    input("Press enter to finish")                      #pause at end so figures can be seen