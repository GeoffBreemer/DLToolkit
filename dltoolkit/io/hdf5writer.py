"""Wrapper for creating HFD5 files"""
import h5py
import os

DATA = "X"
LABELS = "Y"
NAMES = "LABELS"
BUF_SIZE = 10000


class HDF5Writer:
    def __init__(self, dims, output_path, X_key="images", Y_key="labels", bufsize=BUF_SIZE):
        if os.path.exists(output_path):
            raise ValueError("Output path already exists", output_path)

        # Create the root group and X and Y datasets
        self.db = h5py.File(output_path, "w", libver='latest')
        self.X = self.db.create_dataset(X_key, dims, dtype="float")
        self.Y = self.db.create_dataset(Y_key, (dims[0],), dtype="int")

        # Init the buffer
        self.bufsize = bufsize
        self.buffer = {DATA: [], LABELS: []}
        self.index = 0


    def add(self, obs, labels):
        """Add data and flush to disc when the buffer is full"""
        self.buffer[DATA].extend(obs)
        self.buffer[LABELS].extend(labels)

        if len(self.buffer[DATA]) >= self.bufsize:
            self.flush()

    def flush(self):
        """Write to disc and reset the buffer"""
        i = self.index + len(self.buffer[DATA])
        self.X[self.index:i] = self.buffer[DATA]
        self.Y[self.index:i] = self.buffer[LABELS]
        self.index = i
        self.buffer = {DATA: [], LABELS: []}

    def writelabels(self, classlabels):
        """Write the class labels to disc"""
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset(NAMES, (len(classlabels),), dtype=dt)
        labelSet[:] = classlabels

    def close(self):
        """Close the file, flush to disc first if the buffer is not empty"""
        if len(self.buffer[DATA]) > 0:
            self.flush()

        self.db.close()
