"""Simple HDF5 reader that assumes the entire contents can fit in memory. Closes the file right after returning
all the data.
"""
import h5py
import numpy as np


class HDF5Reader:
    def load_hdf5(self, file_path, key):
        with h5py.File(file_path, "r") as f:
            return np.array(f[key][()])
