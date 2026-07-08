# Created 08/07/2026
# Author: Joseph Fields
# Description: [[TODO]]

import pandas as pd

class DataAnalysis():
    def load_file(self, filename):
        # Load the parquet file
        df = pd.read_parquet(filename)

        # View the first 5 rows
        print(df.head())

        # Get a summary of columns, data types, and missing values
        print(df.info())



if __name__ == "__main__":
    analysis = DataAnalysis()
    analysis.load_file('./parquet_files/Segmented_Linear_Baseline.parquet') 