"""Wrapper for creating HFD5 files

Code is based on the excellent book "Deep Learning for Computer Vision" by PyImageSearch available on:
https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/
"""
import h5py, os

# Constants
BUF_FEATURES = "X"
BUF_LABELS = "Y"
DS_CLASS_NAMES = "NAMES"
BUF_SIZE = 10000


class HDF5Writer:
    def __init__(self, dimensions, output_path, feat_key="X", label_key="Y", bufsize=BUF_SIZE):
        """
        Create the new HDF5 file
        :param dimensions: (# of records, # of features)
        :param output_path: full path to the HDF5 file
        :param feat_key: name of the features data set
        :param label_key: name of the labels data set
        :param bufsize: size of the in-memory buffer (= maximum number of records to keep in memory before
        flushing to disc)
        """
        # check if the file already exists, exit if it does
        if os.path.exists(output_path):
            raise ValueError("Output path already exists", output_path)

        # Create the HDF5 file
        self.db = h5py.File(output_path, "w", libver='latest')

        #  Create the two datasets: features and labels
        self.feat_dataset = self.db.create_dataset(feat_key, dimensions, dtype="float")
        self.label_dataset = self.db.create_dataset(label_key, (dimensions[0],), dtype="int")

        # Init the in-memory buffer
        self.buf_size = bufsize
        self.buffer = {BUF_FEATURES: [], BUF_LABELS: []}
        self.index = 0

    def add(self, features, labels):
        """Add features and labels to the buffer and flush it to disc when it is full"""
        self.buffer[BUF_FEATURES].extend(features)
        self.buffer[BUF_LABELS].extend(labels)

        if len(self.buffer[BUF_FEATURES]) >= self.buf_size:
            self.flush()

    def flush(self):
        """Write to disc and reset the buffer"""
        i = self.index + len(self.buffer[BUF_FEATURES])

        # Add buffer contents to the data sets
        self.feat_dataset[self.index:i] = self.buffer[BUF_FEATURES]
        self.label_dataset[self.index:i] = self.buffer[BUF_LABELS]
        self.index = i

        # Reset the buffer
        self.buffer = {BUF_FEATURES: [], BUF_LABELS: []}

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
