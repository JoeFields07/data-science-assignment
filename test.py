import h5py

#mat_file_path = './data/Segmented_Machining_Baseline.mat'
mat_file_path = './data/Segmented_Linear_Baseline.mat'
mat_file_path = './data/Segmented_Spindle5000_Baseline.mat'

with h5py.File(mat_file_path, 'r') as f:
    data_keys = list(f.keys())
    print(data_keys)
    group = f[data_keys[1]]
    
    print("--- THE 18 COLUMNS INSIDE THE STRUCT ---")
    for column_name in group.keys():
        dataset = group[column_name]
        print(f"Column Name: {column_name} | Type: {type(dataset)} | Shape: {dataset.shape if hasattr(dataset, 'shape') else 'N/A'}")