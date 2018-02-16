"""Wrapper for creating HDF5 files

Code is based on the excellent book "Deep Learning for Computer Vision" by PyImageSearch available on:
https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/
"""
import h5py, os
import numpy as np

# Constants
BUF_FEATURES = "X"
BUF_LABELS = "Y"
DS_CLASS_NAMES = "NAMES"
BUF_SIZE = 10000


class HDF5Writer:
    def __init__(self, dimensions, output_path, feat_key="X", label_key="Y", buf_size=BUF_SIZE, del_existing=False):
        """
        Create the new HDF5 file for simple 2D matrices
        :param dimensions: e.g. (# of records, # of features) or (# of images, height, width, # of channels)
        :param output_path: full path to the HDF5 file
        :param feat_key: name of the features data set
        :param label_key: name of the labels data set
        :param buf_size: size of the in-memory buffer (= maximum number of records to keep in memory before
        :param del_existing: delete existing file with the same name True/False
        flushing to disc)
        """
        self.include_labels = True          # Assume target labels are provided

        if del_existing:
            if os.path.exists(output_path):
                os.remove(output_path)
        elif os.path.exists(output_path):
            raise ValueError("Output path already exists", output_path)

        # Create the HDF5 file
        self.db = h5py.File(output_path, "w", libver='latest')

        #  Create the two datasets: features and labels (optional)
        self.feat_dataset = self.db.create_dataset(feat_key, dimensions, dtype="float")

        if label_key is not None:
            self.label_dataset = self.db.create_dataset(label_key, (dimensions[0],), dtype="int")
        else:
            self.include_labels = False

        # Init the in-memory buffer
        self.buf_size = buf_size

        if self.include_labels:
            self.buffer = {BUF_FEATURES: [], BUF_LABELS: []}
        else:
            self.buffer = {BUF_FEATURES: []}

        self.index = 0

    def add(self, features, labels):
        """Add features and labels to the buffer and flush it to disc when it is full"""
        self.buffer[BUF_FEATURES].extend(features)

        if self.include_labels:
            self.buffer[BUF_LABELS].extend(labels)

        if len(self.buffer[BUF_FEATURES]) >= self.buf_size:
            self.flush()

    def flush(self):
        """Write to disc and reset the buffer"""
        i = self.index + len(self.buffer[BUF_FEATURES])

        # Add buffer contents to the data sets
        self.feat_dataset[self.index:i] = self.buffer[BUF_FEATURES]

        if self.include_labels:
            self.label_dataset[self.index:i] = self.buffer[BUF_LABELS]

        self.index = i

        # Reset the buffer
        if self.include_labels:
            self.buffer = {BUF_FEATURES: [], BUF_LABELS: []}
        else:
            self.buffer = {BUF_FEATURES: []}

    def write_class_names(self, class_names):
        """Write the class name strings to disc"""
        dt = h5py.special_dtype(vlen=str)
        class_name_dataset = self.db.create_dataset(DS_CLASS_NAMES, (len(class_names),), dtype=dt)
        class_name_dataset[:] = class_names

    def close(self):
        """Close the file, flush to disc first if the buffer is not empty"""
        if len(self.buffer[BUF_FEATURES]) > 0:
            self.flush()

        self.db.close()

class HDF5Reader:
    """Simple HDF5 reader that assumes the entire contents can fit in memory. Closes the file right after retuning
    the data.
    """
    def load_hdf5(self, file_path, key):
        with h5py.File(file_path, "r") as f:
            return np.array(f[key][()])


# class HDF5Reader_org:
#     _db = None
#
#     def load_hdf5(self, file_path, key):
#         self._db = h5py.File(file_path, "r")
#         data = self._db[key]
#         return np.array(data)
#
#     def close(self):
#         self._db.close()


