import h5py
import numpy as np
import time

start = time.time()

# Function to find non-empty images and return their indices in chunks
def find_non_empty_images_in_chunks(dataset, chunk_size=1000):
    non_empty_indices = []
    total_samples = dataset.shape[0]

    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk = dataset[start:end]

        valid_chunk_indices = np.where(
            (np.sum(chunk, axis=(1, 2, 3)) > 0) &
            (~np.isnan(chunk).any(axis=(1, 2, 3))) &
            (~np.isinf(chunk).any(axis=(1, 2, 3)))
        )[0]

        # Adjust indices relative to the full dataset
        non_empty_indices.extend(start + valid_chunk_indices)

    return np.array(non_empty_indices)


# Function to check for duplicates in chunks (retains only one copy if labels match)
def find_duplicate_images_in_chunks(dataset, labels, chunk_size=1000):
    reshaped_dataset = dataset.reshape(dataset.shape[0], -1)
    unique_images, unique_indices = np.unique(reshaped_dataset, axis=0, return_index=True)

    # Identify duplicates with different labels
    duplicate_indices = []
    for i in range(0, len(reshaped_dataset)):
        if i not in unique_indices:
            matching_index_list = np.where(np.all(reshaped_dataset[unique_indices] == reshaped_dataset[i], axis=1))[0]
            if len(matching_index_list) > 0:
                matching_index = unique_indices[matching_index_list[0]]
                if not np.array_equal(labels[i], labels[matching_index]):
                    duplicate_indices.append(i)

    return unique_indices, duplicate_indices


# Main function to create a new HDF5 file without empty or duplicate images
def create_cleaned_h5_file(input_file_path, output_file_path, chunk_size=1000):
    with h5py.File(input_file_path, 'r') as h5_file:
        labels = h5_file['label'][:]
        sen1 = h5_file['sen1']
        sen2 = h5_file['sen2']

        # Find non-empty images in chunks for sen1 and sen2
        non_empty_indices_sen1 = find_non_empty_images_in_chunks(sen1, chunk_size)
        non_empty_indices_sen2 = find_non_empty_images_in_chunks(sen2, chunk_size)
        non_empty_indices = np.intersect1d(non_empty_indices_sen1, non_empty_indices_sen2)

        # Filter out non-empty images and labels
        filtered_labels = labels[non_empty_indices]

        # Create new datasets in a new HDF5 file to avoid memory issues
        with h5py.File(output_file_path, 'w') as new_h5_file:
            new_h5_file.create_dataset('label', data=filtered_labels)

            sen1_out = new_h5_file.create_dataset('sen1', shape=(len(non_empty_indices), 32, 32, 8), dtype=sen1.dtype)
            sen2_out = new_h5_file.create_dataset('sen2', shape=(len(non_empty_indices), 32, 32, 10), dtype=sen2.dtype)

            for i, idx in enumerate(non_empty_indices):
                sen1_out[i] = sen1[idx]
                sen2_out[i] = sen2[idx]

            # Check for duplicates in chunks and handle them (retaining one with matching labels)
            unique_indices, duplicate_indices = find_duplicate_images_in_chunks(sen1_out[:], filtered_labels)

            if duplicate_indices:
                print(f"Found duplicates with differing labels at positions: {duplicate_indices}")

            print(f"New HDF5 file created at {output_file_path} without empty or duplicate images.")



input_file_path = r"F:\DS_project\m1613658\random\clean_training_1st.h5"
output_file_path = r"F:\DS_project\m1613658\random\clean_training_1st_subset.h5"

# Run the process
create_cleaned_h5_file(input_file_path, output_file_path, chunk_size=20000)
end = time.time()

print(end-start//60)