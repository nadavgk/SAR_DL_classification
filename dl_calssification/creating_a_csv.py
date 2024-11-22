import h5py
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
import time

# Function to get labels for each image
def get_labels(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        labels = h5_file['label'][:]
        label_positions = [np.where(label == 1)[0][0] if np.any(label == 1) else -1 for label in labels]
    return label_positions


# Function to process a chunk of data for Sentinel images
def process_image_chunk(args):
    file_path, start_idx, end_idx, channel, dataset_name, labels = args
    chunk_results = []
    with h5py.File(file_path, 'r') as h5_file:
        dataset = h5_file[dataset_name]
        chunk = dataset[start_idx:end_idx, :, :, channel]
        for i, img in enumerate(chunk):
            image_index = start_idx + i
            col_sum = np.sum(img)
            null_present = np.isnan(img).any() or np.isinf(img).any()
            chunk_results.append({
                'image_index': image_index,
                'channel': channel + 1,
                'sum_of_pixels': col_sum,
                'null_present': null_present,
                'label': labels[image_index]  # Add the corresponding label
            })
    return chunk_results


# Function to parallelize the processing
def process_in_parallel(func, args_list):
    with Pool() as pool:
        results = pool.map(lambda args: func(args), args_list)
    return [item for sublist in results for item in sublist]  # Flatten the results


# Main function to create separate summary tables for `sen-1` and `sen-2`
def create_summary_tables_parallel(input_file_path, sen1_output_csv, sen2_output_csv, chunk_size=1000):
    # Get labels for each image
    labels = get_labels(input_file_path)

    with h5py.File(input_file_path, 'r') as h5_file:
        total_samples = h5_file['label'].shape[0]

    # Prepare argument lists for parallel processing
    sen1_args_list = [(input_file_path, start, min(start + chunk_size, total_samples), channel, 'sen1', labels)
                      for channel in range(8)  # 8 channels in sen1
                      for start in range(0, total_samples, chunk_size)]

    sen2_args_list = [(input_file_path, start, min(start + chunk_size, total_samples), channel, 'sen2', labels)
                      for channel in range(10)  # 10 channels in sen2
                      for start in range(0, total_samples, chunk_size)]

    # Process chunks in parallel for `sen1`
    sen1_results = process_in_parallel(process_image_chunk, sen1_args_list)

    # Process chunks in parallel for `sen2`
    sen2_results = process_in_parallel(process_image_chunk, sen2_args_list)

    # Convert to DataFrames and save to CSV
    sen1_df = pd.DataFrame(sen1_results)
    sen1_df.to_csv(sen1_output_csv, index=False)

    sen2_df = pd.DataFrame(sen2_results)
    sen2_df.to_csv(sen2_output_csv, index=False)

    print(f"Summary tables created and saved to {sen1_output_csv} and {sen2_output_csv}")


if __name__ == "__main__":
    start = time.time()
    # File paths
    input_file_path = r"F:\DS_project\m1613658\random\testing.h5"
    sen1_output_csv = r"F:\DS_project\m1613658\random\training_eda_sen_1.csv"
    sen2_output_csv = r"F:\DS_project\m1613658\random\testing_eda_sen_2.csv"

    # Create the summary tables with parallel processing
    create_summary_tables_parallel(input_file_path, sen1_output_csv, sen2_output_csv, chunk_size=1000)
    end = time.time()
    print((end-start)//60)

