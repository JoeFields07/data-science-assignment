from data_file_helper import FileHelper, ALL_CHANNEL_KEYS, MACH_CHANNEL_KEYS, FEATURE_KEYS
from analysis_helper import AnalysisHelper
import argparse


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
    
    m_baseline = FileHelper('./data/Segmented_Machining_Baseline.mat')
    m_misalign = FileHelper('./data/Segmented_Machining_Misalignment.mat')
    m_cracks = FileHelper('./data/Segmented_Machining_SurfaceCracks.mat')
    m_wear = FileHelper('./data/Segmented_Machining_ToolWear.mat')

    analysis = AnalysisHelper()
    #combine FileHelper dicts into ML friendly matrix
    analysis.import_feature_data([m_baseline.data_stats, m_misalign.data_stats, 
                                  m_cracks.data_stats, m_wear.data_stats], 
                                 MACH_CHANNEL_KEYS, FEATURE_KEYS)
    
    analysis.preprocess_data(0.2, 42)       #split and standardise
    
    analysis.train_PCA()                    #train PCA on X_train
    analysis.apply_PCA()                    #apply trained PCA to X_test
    #plot results
    analysis.plot_PCA(1)


if __name__ =="__main__":
    main()
    input("Press enter to finish")