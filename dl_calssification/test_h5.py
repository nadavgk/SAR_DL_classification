import plotly.express as px
import h5py
import numpy as np
import pandas as pd
import gc

file_path = r"F:\DS_project\m1613658\random\testing_1_perc_subset.h5"
with h5py.File(file_path, 'r') as h5_file:
    print(h5_file.keys())
    labels = h5_file['label'][:,:]
    sen1 = h5_file['sen1'][0,:,:,:]
    sen2 = h5_file['sen2']
    print(labels.shape)
    print(sen1.shape)
    print(sen2.shape)

