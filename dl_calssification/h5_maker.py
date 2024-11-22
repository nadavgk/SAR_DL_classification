import h5py
import numpy as np
import time

start = time.time()

def create_subset_h5_file(input_file_path, output_file_path, num_samples=20000):
    with h5py.File(input_file_path, 'r') as h5_file:
        # Ensure the number of samples doesn't exceed the dataset size
        total_samples = h5_file['sen1'].shape[0]
        num_samples = min(num_samples, total_samples)

        # Randomly select indices for the subset
        selected_indices = np.random.choice(total_samples, num_samples, replace=False)

        # Sort indices to maintain data order
        selected_indices.sort()

        # Extract the subset
        labels_subset = h5_file['label'][selected_indices]
        sen1_subset = h5_file['sen1'][selected_indices]
        sen2_subset = h5_file['sen2'][selected_indices]

        # Create the new HDF5 file with the subset
        with h5py.File(output_file_path, 'w') as new_h5_file:
            new_h5_file.create_dataset('label', data=labels_subset)
            new_h5_file.create_dataset('sen1', data=sen1_subset)
            new_h5_file.create_dataset('sen2', data=sen2_subset)

        print(f"Subset HDF5 file created at {output_file_path} with {num_samples} samples.")


# File paths
input_file_path = r"F:\DS_project\m1613658\random\training.h5"
output_file_path = r"F:\DS_project\m1613658\random\clean_training_1st.h5"


# Create the subset
create_subset_h5_file(input_file_path, output_file_path, num_samples=20000)
end = time.time()

print(end-start//60)