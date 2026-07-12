from data_file_helper import FileHelper, ALL_CHANNEL_KEYS, MACH_CHANNEL_KEYS, FEATURE_KEYS, DATA_FOLDER
from analysis_helper import AnalysisHelper
import argparse

#TODO
    #add a simple classifier?
    #command line interface
        #either a preset that loads linear, machining or spindles (separate and both)
        #groups of filenames, throws a controlled error if the features use different keys

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

    #either all need to be Machining or none
    

    for filename in args.file:
        data_obj_list[c] = FileHelper(args.directory + filename)    #could avoid saving the data
        stats_list[c] = data_obj_list[c].data_stats
        labels[c] = data_obj_list[c].experiment + data_obj_list[c].variant
        experiment_list[c] = data_obj_list[c].experiment
        c+=1

    valid_file_combination = False

    if any(experiment_list) == "Machining":     #if Machining is present, all files need to be Machining
        if all(experiment_list) == "Machining":
            valid_file_combination == True
    
    else:
        valid_file_combination == True

    if valid_file_combination:
        channel_keys = data_obj_list[0].channel_keys

        #m_baseline = FileHelper('./data/Segmented_Machining_Baseline.mat')
        #m_misalign = FileHelper('./data/Segmented_Machining_Misalignment.mat')
        #m_cracks = FileHelper('./data/Segmented_Machining_SurfaceCracks.mat')
        #m_wear = FileHelper('./data/Segmented_Machining_ToolWear.mat')
        #l_base = FileHelper('./data/Segmented_Linear_Baseline.mat')
        #l_heavy = FileHelper('./data/Segmented_Linear_Heavy.mat')
        #l_ovr = FileHelper('./data/Segmented_Linear_Override.mat')
        
        analysis = AnalysisHelper()
        #combine FileHelper dicts into ML friendly matrix
        analysis.import_feature_data(stats_list, channel_keys, FEATURE_KEYS)
        
        analysis.remove_outliers(4)
        analysis.preprocess_data(0.01, 42)       #split and standardise
        
        analysis.train_PCA()                    #train PCA on X_train
        analysis.apply_PCA()                    #apply trained PCA to X_test
        
        analysis.plot_PCA(1, labels)            #plot results
        analysis.plot_feature(2, "SpindleAccX_p2p", "SpindleAccY_kurtosis", "SpindleAccX_mean", labels)



if __name__ =="__main__":
    main()
    input("Press enter to finish")