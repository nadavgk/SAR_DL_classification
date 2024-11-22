import h5py
import numpy as np

# File paths
original_file_path = r"F:\DS_project\m1613658\random\training.h5"
subset_file_path = r"F:\DS_project\m1613658\random\training_1_perc_subset.h5"

# Open the original file and sample 1% of data
with h5py.File(original_file_path, 'r') as original_file:
    # Load datasets
    labels = original_file['label'][:]
    sen1 = original_file['sen1'][:]
    sen2 = original_file['sen2'][:]

    # Calculate 1% sample size
    sample_size = int(0.01 * len(labels))

    # Randomly select indices
    indices = np.random.choice(len(labels), sample_size, replace=False)

    # Extract subsets
    labels_subset = labels[indices]
    sen1_subset = sen1[indices]
    sen2_subset = sen2[indices]

    # Save subsets to a new HDF5 file
    with h5py.File(subset_file_path, 'w') as subset_file:
        subset_file.create_dataset('label', data=labels_subset)
        subset_file.create_dataset('sen1', data=sen1_subset)
        subset_file.create_dataset('sen2', data=sen2_subset)

print("Subset HDF5 file created successfully!")
