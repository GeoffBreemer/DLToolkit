"""Image generator using a HDF5 file as the source

Code is based on the excellent book "Deep Learning for Computer Vision" by PyImageSearch available on:
https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/
"""
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py


class HDF5Generator:
    def __init__(self, dbpath, batch_size, preprocessors=None, augment=None, onehot=False,
                 num_classes=2, label_key="Y"):
        self._batch_size = batch_size
        self._preprocessors = preprocessors
        self._augment = augment
        self._onehot = onehot
        self._num_classes = num_classes

        # Open the database
        self._db = h5py.File(dbpath, "r")
        self._num_images = self._db[label_key].shape[0]

    def generator(self, num_epochs=np.inf, feat_key="X", label_key="Y"):
        """Generate batches of data"""
        epochs = 0

        while epochs < num_epochs:
            for i in np.arange(0, self._num_images, self._batch_size):
                # Get the current batch
                X = self._db[feat_key][i:i + self._batch_size]
                Y = self._db[label_key][i:i + self._batch_size]

                # One-hot encode
                if self._onehot:
                    Y = to_categorical(Y, self._num_classes)

                # Apply preprocessors
                if self._preprocessors is not None:
                    processed_images = []

                    for image in X:
                        for p in self._preprocessors:
                            image = p.preprocess(image)

                        processed_images.append(image)

                    X = np.array(processed_images)

                # Apply augmentation
                if self._augment is not None:
                    (X, Y) = next(self._augment.flow(X, Y, batch_size=self._batch_size))

                # Return
                yield (X, Y)

            epochs +=1

    def close(self):
        self._db.close()


class HDF5Generator_Segment:
    def __init__(self, image_db_path, mask_db_path, batch_size, num_classes, data_gen_args=None, augment=None, label_key="Y"):
        self._batch_size = batch_size
        self._augment = augment

        # Open the database
        self._db_image = h5py.File(image_db_path, "r")
        self._db_mask = h5py.File(mask_db_path, "r")
        self._num_images = self._db_image[label_key].shape[0]
        self._num_classes = num_classes

        self.data_gen_args = data_gen_args

        self.image_datagen = ImageDataGenerator(**data_gen_args)
        self.mask_datagen = ImageDataGenerator(**data_gen_args)


    def generator(self, converter, num_epochs=np.inf, feat_key="X"):
        """Generate batches of data"""
        epochs = 0
        RANDOM_STATE = 122177

        while epochs < num_epochs:
            for i in np.arange(0, self._num_images, self._batch_size):
                # Get the current batch
                imgs = self._db[feat_key][i:i + self._batch_size]
                masks = self._db[feat_key][i:i + self._batch_size]

                # Apply augmentation
                # train_image_gen = self.image_datagen.flow(imgs, batch_size=self._batch_size, shuffle=True, seed=RANDOM_STATE)
                # train_mask_gen = self.mask_datagen.flow(masks, batch_size=self._batch_size, shuffle=True, seed=RANDOM_STATE)

                # the_generator = zip(train_image_gen, train_mask_gen)
                # (imgs, masks) = next(the_generator)

                # Convert masks to UNet format
                # TODO TODO TODO TODO
                masks = converter(masks, self._num_classes )

                # Return
                yield (imgs, masks)

            epochs +=1

    def close(self):
        self._db.close()
