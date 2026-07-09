from data_file_helper import FileHelper
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
    
    machining_baseline = FileHelper('./data/Segmented_Machining_Baseline.mat')
    machining_misalign = FileHelper('./data/Segmented_Machining_Misalignment.mat')
    machining_cracks = FileHelper('./data/Segmented_Machining_SurfaceCracks.mat')
    machining_wear = FileHelper('./data/Segmented_Machining_ToolWear.mat')

    #need to combine the features and turn them into some kind of labelled dataset

    #standardise

    #PCA

    #plot results



if __name__ =="__main__":
    main()