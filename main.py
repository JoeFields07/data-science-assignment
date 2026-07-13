# Created 08/07/2026
# Author: Joseph Fields
# Description: [[TODO]]

from data_file_helper import FileHelper, ALL_CHANNEL_KEYS, MACH_CHANNEL_KEYS, FEATURE_KEYS
from analysis_helper import AnalysisHelper
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process a specific data file.")
    
    parser.add_argument(
        "-d", "--directory",
        type=str,
        required=False,
        default="./data/",
        help="Folder containing cut files (default is './data/')"
    )

    parser.add_argument(        #returns none if nothing entered
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
    
    # 3. Parse the arguments from the terminal
    args = parser.parse_args() 

    
    data_obj_list = []
    experiment_list = []
    c = 0

    for filename in args.file:
        data_obj_list.append(FileHelper(args.directory + filename))    #could avoid saving the data
        experiment_list.append(data_obj_list[c].experiment)
        c+=1

    valid_file_combination = False

    #Valid if either "Machining" isn't in list OR if "Machining" is only thing in list
    valid_file_combination = ("Machining" not in experiment_list) or (set(experiment_list) == {"Machining"})

    if valid_file_combination:
        stats_list = []
        labels = []
        for obj in data_obj_list:       #loop through list of file helper objects
            
            if obj.feature_filepath.is_file():              #if feature data exists, load and don't calculate again
                print("Feature file already exists") if args.verbose else 0
                obj.data_stats = obj.load_feature_file()

            else:
                if obj.data_filepath.is_file():         #if feature file doesn't exist, calculate it
                    print("Feature file does not exist") if args.verbose else 0
                    obj.data = obj.load_data_file()
                    obj.data_stats = obj.extract_features()
                    obj.export_features()               #save newly extracted features
        
                else:
                    FileNotFoundError("Error: Invalid data file name")
                    
            stats_list.append(obj.data_stats)
            labels.append(obj.experiment + obj.variant)
        
        channel_keys = data_obj_list[0].channel_keys
        
        analysis = AnalysisHelper()
        #combine FileHelper dicts into ML friendly matrix
        analysis.import_feature_data(stats_list, channel_keys, FEATURE_KEYS, 6.0)
        
        analysis.preprocess_data(0.2, 42)       #split and standardise
        
        analysis.train_PCA()                    #train PCA on X_train
        analysis.apply_PCA()                    #apply trained PCA to X_test
        
        analysis.plot_PCA(1, labels)            #plot results
        analysis.plot_feature(2, "SpindleAccX_p2p", "SpindleAccY_kurtosis", "SpindleAccX_mean", labels)
        analysis.train_classifier(classifier='RF', C=None, n_estimators=100, random_state=42)
        analysis.predict_classifier(3, labels)

        pass
    else:
        print("Error: Invalid combination of data files")



if __name__ =="__main__":
    main()
    input("Press enter to finish")