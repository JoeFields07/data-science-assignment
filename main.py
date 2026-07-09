
from data_file_handler import FileHandler
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
    
    #machining_baseline = FileHandler('./data/Segmented_Machining_Baseline.mat')
    #machining_misalign = FileHandler('./data/Segmented_Machining_Misalignment.mat')
    #machining_cracks = FileHandler('./data/Segmented_Machining_SurfaceCracks.mat')
    #machining_wear = FileHandler('./data/Segmented_Machining_ToolWear.mat')
    t = FileHandler('./data/Segmented_Spindle5000_Baseline.mat')
    t = FileHandler('./data/Segmented_Spindle5000_Heavy.mat')
    t = FileHandler('./data/Segmented_Spindle5000_Override.mat')
    t = FileHandler('./data/Segmented_Spindle5000_Unbalanced.mat')
    t = FileHandler('./data/Segmented_Spindle12000_Baseline.mat')
    t = FileHandler('./data/Segmented_Spindle12000_Heavy.mat')
    t = FileHandler('./data/Segmented_Spindle12000_Override.mat')
    t = FileHandler('./data/Segmented_Spindle12000_Unbalanced.mat')

    #need to combine the features and turn them into some kind of labelled dataset

    #standardise

    #PCA

    #plot results

    print('test')


if __name__ =="__main__":
    main()