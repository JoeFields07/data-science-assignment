from data_file_helper import FileHelper, ALL_CHANNEL_KEYS, MACH_CHANNEL_KEYS, FEATURE_KEYS
from analysis_helper import AnalysisHelper
import argparse

#TODO
#add a simple classifier?
def main():
    parser = argparse.ArgumentParser(description="Process a specific data file.")
    
    # 2. Define the argument you want to accept (e.g., --file or -f)
    parser.add_argument(
        "-f", "--file", 
        type=str, 
        required=False, 
        help="The path of the .mat file for analysis"
    )
    
    # 3. Parse the arguments from the terminal
    args = parser.parse_args()
    analysis = AnalysisHelper()

    #m_baseline = FileHelper('./data/Segmented_Machining_Baseline.mat')
    #m_misalign = FileHelper('./data/Segmented_Machining_Misalignment.mat')
    #m_cracks = FileHelper('./data/Segmented_Machining_SurfaceCracks.mat')
    #m_wear = FileHelper('./data/Segmented_Machining_ToolWear.mat')
    l_base = FileHelper('./data/Segmented_Linear_Baseline.mat')
    l_heavy = FileHelper('./data/Segmented_Linear_Heavy.mat')
    l_ovr = FileHelper('./data/Segmented_Linear_Override.mat')
    data_list = [l_base.data_stats, l_heavy.data_stats, l_ovr.data_stats]
    
    #combine FileHelper dicts into ML friendly matrix
    analysis.import_feature_data(data_list, ALL_CHANNEL_KEYS, FEATURE_KEYS)
    
    analysis.remove_outliers(4)
    analysis.preprocess_data(0.01, 42)       #split and standardise
    
    analysis.train_PCA()                    #train PCA on X_train
    analysis.apply_PCA()                    #apply trained PCA to X_test
    #plot results
    #analysis.plot_PCA(1, ["Baseline", "Misalign", "Cracks", "Wear"])
    analysis.plot_PCA(1, ["Baseline", "Heavy", "Override"])
    analysis.plot_feature(2, "SpindleAccX_p2p", "SpindleAccY_kurtosis", "SpindleAccX_mean", ["Baseline", "Heavy", "Override"])

if __name__ =="__main__":
    main()
    input("Press enter to finish")