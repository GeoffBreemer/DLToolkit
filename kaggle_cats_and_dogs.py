"""Train AlexNet on the Kaggle Cats & Dogs data set"""
import json
import os
import numpy as np
import progressbar
import cv2
import matplotlib
matplotlib.use("Agg")

from settings import settings_cats_and_dogs as settings

from dltoolkit.preprocess import ResizeWithAspectRatioPreprocessor, ImgToArrayPreprocessor, ResizePreprocessor, PatchPreprocessor, SubtractMeansPreprocessor
from dltoolkit.io import HDF5Generator, HDF5Writer
from dltoolkit.nn import AlexNetNN
from dltoolkit.utils import TrainingMonitor, ranked_accuracy, save_model_architecture

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import print_summary
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


TRAIN_SET = "TRAIN_SET"
VAL_SET = "VAL_SET"
TEST_SET = "TEST_SET"


def create_hdf5():
    """Convert the data set to HDF5 format"""
    from imutils import paths

    # Load image paths and determine class label from the file name
    train_paths = list(paths.list_images(settings.DATA_PATH))
    train_labels = [p.split(os.path.sep)[2].split(".")[0] for p in train_paths]

    # Encode labels
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)

    # Split the data into a training, validation and test set
    (train_paths, test_paths, train_labels, test_labels) = train_test_split(train_paths, train_labels,
                                                                            test_size=settings.NUM_TEST_IMAGES,
                                                                            stratify=train_labels,
                                                                            random_state=settings.RANDOM_STATE)

    (train_paths, val_paths, train_labels, val_labels) = train_test_split(train_paths, train_labels,
                                                                          test_size=settings.NUM_VAL_IMAGES,
                                                                          stratify=train_labels,
                                                                          random_state=settings.RANDOM_STATE)

    # Create list of all data sets and their paths
    data = [(TRAIN_SET, train_paths, train_labels, settings.TRAIN_SET_HFD5_PATH),
            (VAL_SET, val_paths, val_labels, settings.VAL_SET_HFD5_PATH),
            (TEST_SET, test_paths, test_labels, settings.TEST_SET_HFD5_PATH)]

    # Init image resizer and channel averages
    aspect_process = ResizeWithAspectRatioPreprocessor(settings.IMG_DIM_WIDTH, settings.IMG_DIM_HEIGHT)
    (R_vals, G_vals, B_vals) = ([], [], [])

    # Write each dataset to HDF5
    for (dt, paths, labels, output_path) in data:
        writer = HDF5Writer((len(paths), settings.IMG_DIM_WIDTH,
                             settings.IMG_DIM_HEIGHT, settings.IMG_CHANNELS), output_path)

        # Prepare progress bar
        widgets = ["Creating HFD5 database ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

        # Preprocess each image and write to hfd5. Keep track of mean RGB values for the training set
        for (i, (path, label)) in enumerate(zip(paths, labels)):
            image = cv2.imread(path)
            image = aspect_process.preprocess(image)

            if dt == TRAIN_SET:
                (b, g, r) = cv2.mean(image)[:3]
                R_vals.append(r)
                G_vals.append(g)
                B_vals.append(b)

            writer.add([image], [label])
            pbar.update(i)

        pbar.finish()
        writer.close()

    # Save RGB means to disk
    D = {"R": np.mean(R_vals), "G" : np.mean(G_vals), "B": np.mean(B_vals)}
    f = open(settings.RGB_MEANS_PATH, "w")
    f.write(json.dumps(D))
    f.close()


def create_model():
    """Create and compile the AlexNet model"""
    opt = Adam(lr=settings.ADAM_LR)
    model = AlexNetNN(num_classes=settings.NUM_CLASSES).build_model()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def train_alexnet(model):
    """Train the model"""
    # Prepare data augmenter
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.15,
                             horizontal_flip=True, fill_mode="nearest")

    # Load RGB means
    means = json.loads(open(settings.RGB_MEANS_PATH).read())

    # Init preprocessors
    res_pre = ResizePreprocessor(AlexNetNN._img_width, AlexNetNN._img_height)
    patch_pre = PatchPreprocessor(AlexNetNN._img_width, AlexNetNN._img_height)
    mean_pre = SubtractMeansPreprocessor(means["R"], means["G"], means["B"])
    itoa_pre = ImgToArrayPreprocessor()

    # Init data generators
    train_gen = HDF5Generator(settings.TRAIN_SET_HFD5_PATH, batchsize=settings.BATCH_SIZE, augment=aug,
                              preprocessors=[patch_pre, mean_pre, itoa_pre], num_classes=settings.NUM_CLASSES)

    val_gen = HDF5Generator(settings.VAL_SET_HFD5_PATH, batchsize=settings.BATCH_SIZE, augment=aug,
                            preprocessors=[res_pre, mean_pre, itoa_pre], num_classes=settings.NUM_CLASSES)

    # Train the model
    path = os.path.sep.join([settings.OUTPUT_PATH, "{}.png".format(os.getpid())])
    callbacks = [TrainingMonitor(fig_path=path, json_path=settings.HISTORY_PATH),
                 ModelCheckpoint(settings.MODEL_PATH, monitor="val_loss", mode="min", save_best_only=True, verbose=1)]

    print("--> Training the model...")
    model.fit_generator(train_gen.generator(),
                        steps_per_epoch=train_gen._num_images // settings.BATCH_SIZE,
                        validation_data=val_gen.generator(),
                        validation_steps=val_gen._num_images // settings.BATCH_SIZE,
                        epochs=settings.NUM_EPOCHS,
                        max_queue_size=settings.BATCH_SIZE * 2,
                        callbacks=callbacks,
                        verbose=1)

    # Save the last model
    print("--> Saving the trained model...")
    model.save(settings.MODEL_PATH, overwrite=True)

    train_gen.close()
    val_gen.close()


def visualise_model(model):
    # Plot model to the console
    print_summary(model)

    # Write the diagram to disc
    save_model_architecture(model, settings.OUTPUT_PATH + "/catsdogs_alexnet", show_shapes=True)


def evaluate_model():
    """test the model on the unseen test set"""
    # Load RGB means
    means = json.loads(open(settings.RGB_MEANS_PATH).read())

    # Init preprocessors
    res_pre = ResizePreprocessor(AlexNetNN._img_width, AlexNetNN._img_height)
    mean_pre = SubtractMeansPreprocessor(means["R"], means["G"], means["B"])
    itoa_pre = ImgToArrayPreprocessor()

    # Load the saved model
    model = load_model(settings.MODEL_PATH)

    # Create the data generator
    test_gen = HDF5Generator(settings.TEST_SET_HFD5_PATH, batchsize=settings.BATCH_SIZE,
                             preprocessors=[res_pre, mean_pre, itoa_pre], num_classes=settings.NUM_CLASSES)

    # Make predictions
    print("--> Evaluating the model...")
    predictions = model.predict_generator(test_gen.generator(),
                                          steps=test_gen._num_images // settings.BATCH_SIZE,
                                          max_queue_size=settings.BATCH_SIZE * 2)

    # Print results
    (rank1, _) = ranked_accuracy(predictions, test_gen._db["Y"])
    print("Accuracy: {:.2f}%".format(rank1 * 100))

    test_gen.close()


if __name__ == "__main__":
    # Convert the Kaggle dataset to HDF5 format
    # create_hdf5()

    # Create the model
    # model = create_model()

    # Visualise the model
    # visualise_model(model)

    # Train the model on the training and validation sets
    # train_alexnet(model)

    # Evaluate it on the test set
    evaluate_model()
