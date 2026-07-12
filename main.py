#python main.py -f Segmented_Linear_Baseline.mat Segmented_Linear_Heavy.mat Segmented_Linear_Override.mat Segmented_Spindle5000_Baseline.mat Segmented_Spindle5000_Heavy.mat Segmented_Spindle5000_Override.mat Segmented_Spindle5000_Unbalanced.mat -v

from data_file_helper import FileHelper, ALL_CHANNEL_KEYS, MACH_CHANNEL_KEYS, FEATURE_KEYS
from analysis_helper import AnalysisHelper
import argparse

#TODO
    #add a simple classifier?

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

    experiment_list = []
    data_obj_list = []
    stats_list = []
    labels = []
    c = 0

    for filename in args.file:
        data_obj_list.append(FileHelper(args.directory + filename))    #could avoid saving the data
        stats_list.append(data_obj_list[c].data_stats)
        labels.append(data_obj_list[c].experiment + data_obj_list[c].variant)
        experiment_list.append(data_obj_list[c].experiment)
        c+=1

    valid_file_combination = False

    #Valid if either "Machining" isn't in list OR if "Machining" is only thing in list
    valid_file_combination = ("Machining" not in experiment_list) or (set(experiment_list) == {"Machining"})

    if valid_file_combination:
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

    else:
        print("Error: Invalid combination of data files")



if __name__ =="__main__":
    main()
    input("Press enter to finish")