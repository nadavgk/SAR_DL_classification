import h5py
import numpy as np

# Replace 'path/to/your/file.h5' with your .h5 file path
file_path = r"F:\DS_project\m1613658\random\training.h5"

# Open the HDF5 file and check the data
with h5py.File(file_path, 'r') as h5_file:
    # Load the data
    labels = h5_file['label'][:]
    sen1 = h5_file['sen1'][:]
    sen2 = h5_file['sen2'][:]

    # Check for non-empty images in sen1 and sen2
    sen1_non_empty = np.all(np.sum(sen1, axis=(1, 2, 3)) > 0)
    sen2_non_empty = np.all(np.sum(sen2, axis=(1, 2, 3)) > 0)

    # Check if all labels are non-zero
    all_non_zero_labels = np.all(labels != 0)

    # Display the results
    if sen1_non_empty:
        print("All sen1 images are non-empty.")
    else:
        print("Some sen1 images are empty.")

    if sen2_non_empty:
        print("All sen2 images are non-empty.")
    else:
        print("Some sen2 images are empty.")

    if all_non_zero_labels:
        print("All labels are non-zero.")
    else:
        print("Some labels have zero values.")

import h5py
import numpy as np

# Replace 'path/to/your/file.h5' with your .h5 file path
file_path = 'path/to/your/file.h5'


# Function to check for duplicate images
def check_for_duplicates(dataset, dataset_name):
    # Reshape the dataset to 2D for easier comparison (flatten each 3D image)
    reshaped_dataset = dataset.reshape(dataset.shape[0], -1)

    # Create a set of unique images
    unique_images, indices = np.unique(reshaped_dataset, axis=0, return_index=True)

    if len(unique_images) == dataset.shape[0]:
        print(f"No duplicate images found in {dataset_name}.")
    else:
        print(f"Duplicate images found in {dataset_name}.")
        duplicates_count = dataset.shape[0] - len(unique_images)
        print(f"Number of duplicate images: {duplicates_count}")


# Open the HDF5 file and check the data
with h5py.File(file_path, 'r') as h5_file:
    sen1 = h5_file['sen1'][:]
    sen2 = h5_file['sen2'][:]

    # Check for duplicate images in sen1 and sen2
    check_for_duplicates(sen1, 'sen1')
    check_for_duplicates(sen2, 'sen2')