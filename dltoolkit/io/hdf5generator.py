"""Image generator using a HDF5 file as the source"""
from keras.utils import to_categorical
import numpy as np
import h5py


class HDF5Generator:
    def __init__(self, dbpath, batchsize, preprocessors=None, augment=None, onehot=True,
                 num_classes=2, Y_key="labels"):
        self.batchsize = batchsize
        self.preprocessors = preprocessors
        self.augment = augment
        self.onehot = onehot
        self.num_classes = num_classes

        # Open the database
        self.db = h5py.File(dbpath, "r")
        self.num_images = self.db[Y_key].shape[0]

    def generator(self, num_epochs=np.inf, X_key="images", Y_key="labels"):
        epochs = 0

        while epochs < num_epochs:
            for i in np.arange(0, self.num_images, self.batchsize):
                # Get the current batch
                X = self.db[X_key][i:i+self.batchsize]
                Y = self.db[Y_key][i:i+self.batchsize]

                # One-hot encode
                if self.onehot:
                    Y = to_categorical(Y, self.num_classes)

                # Apply preprocessors
                if self.preprocessors is not None:
                    processed_images = []

                    for image in X:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        processed_images.append(image)

                    X = np.array(processed_images)

                # Apply augmentation
                if self.augment is not None:
                    (X, Y) = next(self.augment.flow(X, Y, batch_size=self.batchsize))

                # Return
                yield (X, Y)

            epochs +=1

    def close(self):
        self.db.close()