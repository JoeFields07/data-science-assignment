from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_file_handler import FileHandler
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process a specific data file.")
    
    # 2. Define the argument you want to accept (e.g., --file or -f)
    parser.add_init_argument(
        "-f", "--file", 
        type=str, 
        required=False, 
        help="The path of the .mat file for analysis"
    )
    
    # 3. Parse the arguments from the terminal
    args = parser.parse_args()
    if 
    machining_baseline = FileHandler('./data/Segmented_Machining_Baseline.mat')
    machining_misalign = FileHandler('./data/Segmented_Machining_Misalignment.mat')
    machining_cracks = FileHandler('./data/Segmented_Machining_SurfaceCracks.mat')
    machining_wear = FileHandler('./data/Segmented_Machining_ToolWear.mat')

    #need to combine the features and turn them into some kind of labelled dataset

    #standardise

    #PCA

    #plot results

    print('test')


if __name__ =="__main__":
    main()