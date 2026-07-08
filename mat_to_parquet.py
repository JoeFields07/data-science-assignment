from pathlib import Path
import h5py
import pandas as pd
import numpy as np

# Define your directories
input_dir = Path("./data")
output_dir = Path("./data")
output_dir.mkdir(exist_ok=True)

# Batch process every v7.3 .mat file
for mat_path in input_dir.glob("*.mat"):
    try:
        # Open the file using h5py
        with h5py.File(mat_path, 'r') as f:
            clean_data = {}
            
            # Loop through the variables stored in the MAT file
            for key in f.keys():
                # Skip MATLAB system metadata keys
                if key.startswith('#'):
                    continue
                
                # Get the dataset
                dataset = f[key]
                
                # If it's a standard array/matrix, read and transpose it (.T)
                # MATLAB stores column-major, HDF5/Python stores row-major
                if isinstance(dataset, h5py.Dataset):
                    data_array = np.array(dataset).T
                    
                    # Flatten it to 1D if it's a single column/row for the DataFrame
                    clean_data[key] = data_array.flatten()
            
            # Convert to DataFrame and save to Parquet
            df = pd.DataFrame(clean_data)
            
            parquet_path = output_dir / f"{mat_path.stem}.parquet"
            df.to_parquet(parquet_path, engine='pyarrow', index=False)
            print(f"Successfully converted v7.3 file: {mat_path.name}")
            
    except Exception as e:
        print(f"Failed to convert {mat_path.name}: {e}")