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
    """Generator specifically for semantic segmentation data, i.e. images and ground truth images"""
    def __init__(self, image_db_path, mask_db_path, batch_size, num_classes, converter=None, data_gen_args=None, feat_key="X"):
        self._batch_size = batch_size

        # Open the database
        self._db_image = h5py.File(image_db_path, "r")
        self._db_mask = h5py.File(mask_db_path, "r")

        # Create data generators if parameters were provided
        self.data_gen_args = data_gen_args
        if not data_gen_args is None:
            self.image_datagen = ImageDataGenerator(**data_gen_args)
            self.mask_datagen = ImageDataGenerator(**data_gen_args)
        else:
            self.image_datagen = None
            self.mask_datagen = None

        self._num_classes = num_classes
        self._num_images = self._db_image[feat_key].shape[0]
        self._feat_key = feat_key
        self._converter = converter
        self.img_shape = self._db_image[feat_key].shape

    def num_images(self):
        return self._num_images

    def generator(self, num_epochs=np.inf, dim_reorder=None):
        """Generate batches of data"""
        epochs = 0
        RANDOM_STATE = 42

        while epochs < num_epochs:
            for i in np.arange(0, self._num_images, self._batch_size):
                # Get the current batch
                imgs = self._db_image[self._feat_key][i:i + self._batch_size]
                masks = self._db_mask[self._feat_key][i:i + self._batch_size]

                # Apply augmentation
                if not self.image_datagen is None:
                    seed = RANDOM_STATE*epochs

                    imgs = next(self.image_datagen.flow(imgs, batch_size=self._batch_size, shuffle=True, seed=seed))
                    masks = next(self.mask_datagen.flow(masks, batch_size=self._batch_size, shuffle=True, seed=seed))

                # Convert masks to the format produced by the segmentation model
                if not self._converter is None:
                    masks = self._converter(masks, self._num_classes)

                if dim_reorder is not None:
                    # Reorder the dimensions to what the 3D Unet expects
                    imgs = np.transpose(imgs, axes=dim_reorder) #(0, 2, 3, 1, 4))
                    masks = np.transpose(masks, axes=dim_reorder) #(0, 2, 3, 1, 4))

                # Return
                yield (imgs, masks)

            epochs +=1

    def close(self):
        self._db.close()
