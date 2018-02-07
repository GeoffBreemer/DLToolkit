import cv2, re, argparse
import os, sys, dicom
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from itertools import zip_longest

from dltoolkit.nn.segment import FCN_NN, jaccard_coef, dice_coef, dice_coef_loss
from dltoolkit.utils.generic import str2bool

RANDOM_STATE = 122177
NUM_EPOCH = 1  # 10
MINI_BATCH_SIZE = 1

MODEL_PATH = "../../savedmodels/"
OUTPUT_PATH = "../../output/"
DATASET_NAME = "rvsc"

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'P(\d{02})-(\d{04})-.*', ctr_path)
        self.patient_no = match.group(1)
        self.img_no = match.group(2)


def center_crop(ndarray, crop_size):
    """Input ndarray is of rank 3 (height, width, depth).
    Argument crop_size is an integer for square cropping only.
    Performs padding and center cropping to a specified size.
    """
    h, w, d = ndarray.shape
    if crop_size == 0:
        raise ValueError('argument crop_size must be non-zero integer')

    if any([dim < crop_size for dim in (h, w)]):
        # zero pad along each (h, w) dimension before center cropping
        pad_h = (crop_size - h) if (h < crop_size) else 0
        pad_w = (crop_size - w) if (w < crop_size) else 0
        rem_h = pad_h % 2
        rem_w = pad_w % 2
        pad_dim_h = (pad_h // 2, pad_h // 2 + rem_h)
        pad_dim_w = (pad_w // 2, pad_w // 2 + rem_w)
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, pad_dim_w, (0, 0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
        h, w, d = ndarray.shape

    # center crop
    h_offset = (h - crop_size) // 2
    w_offset = (w - crop_size) // 2
    cropped = ndarray[h_offset:(h_offset + crop_size),
              w_offset:(w_offset + crop_size), :]

    return cropped


def lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5):
    lrate = base_lr * (1.0 - (curr_iter // float(max_iter))) ** power
    K.set_value(model.optimizer.lr, lrate)

    return K.eval(model.optimizer.lr)


def read_contour(contour, data_path, return_mask=True):
    img_path = [dirpath for dirpath, dirnames, files in os.walk(data_path)
                if contour.patient_no + 'dicom' in dirpath][0]
    filename = 'P{:s}-{:s}.dcm'.format(contour.patient_no, contour.img_no)
    full_path = os.path.join(img_path, filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype('int')

    if img.ndim < 3:
        img = img[..., np.newaxis]

    if not return_mask:
        return img, None

    mask = np.zeros_like(img, dtype='uint8')
    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)

    if mask.ndim < 3:
        mask = mask[..., np.newaxis]

    return img, mask


def map_all_contours(data_path, contour_type, shuffle=True):
    list_files = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in files if 'list' in f]
    contours = []
    for f in list_files:
        for line in open(f).readlines():
            line = line.strip().replace('\\', '/')
            full_path = os.path.join(data_path, line)
            if contour_type + 'contour' in full_path:
                contours.append(full_path)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)

    print('Number of examples: {:d}'.format(len(contours)))

    contours = map(Contour, contours)

    return contours


def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))

    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path, return_mask=True)
        img = center_crop(img, crop_size=crop_size)
        # img = img.astype("float") / 255.0    # GB
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask

    return images, masks


if __name__ == '__main__':
    np.random.seed(RANDOM_STATE)

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to the data set")
    ap.add_argument("-l", "--load", type=str2bool, nargs='?',
                    const=True, required=False, help="Set to True to load a previously trained model")
    ap.add_argument("-c", "--contour", required=True, help="Contour type (i/o/myo)")
    args = vars(ap.parse_args())

    contour_type = args["contour"]
    crop_size = 200

    print('Mapping ground truth ' + contour_type + ' contours to images in train...')
    train_ctrs = map_all_contours(args["dataset"], contour_type, shuffle=True)
    print('Done mapping training set')
    train_ctrs = list(train_ctrs)       # GB

    split = int(0.1 * len(train_ctrs))
    dev_ctrs = train_ctrs[0:split]
    train_ctrs = train_ctrs[split:]

    print('\nBuilding train dataset ...')
    img_train, mask_train = export_all_contours(train_ctrs,
                                                args["dataset"],
                                                crop_size=crop_size)
    print('\nBuilding dev dataset ...')
    img_dev, mask_dev = export_all_contours(dev_ctrs,
                                            args["dataset"],
                                            crop_size=crop_size)

    input_shape = (crop_size, crop_size, 1)
    num_classes = 2

    nnarch = FCN_NN(crop_size, crop_size, 1, num_classes)
    model = nnarch.build_model()
    model.summary()
    model.compile(optimizer=nnarch.optimizer, loss=dice_coef_loss,
                         metrics=['accuracy', dice_coef, jaccard_coef])
                         # metrics = ['accuracy'])

    kwargs = dict(
        rotation_range=0,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=False,
        vertical_flip=False,
    )

    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                         batch_size=MINI_BATCH_SIZE, seed=RANDOM_STATE)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                       batch_size=MINI_BATCH_SIZE, seed=RANDOM_STATE)
    train_generator = zip_longest(image_generator, mask_generator)

    max_iter = (len(train_ctrs) // MINI_BATCH_SIZE) * NUM_EPOCH
    curr_iter = 0
    base_lr = K.eval(nnarch.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)

    for e in range(NUM_EPOCH):
        print('\nMain Epoch {:d}\n'.format(e + 1))
        print('\nLearning rate: {:6f}\n'.format(lrate))

        train_result = []
        for iteration in range(len(img_train) // MINI_BATCH_SIZE):
            img, mask = next(train_generator)
            res = model.train_on_batch(img, mask)

            curr_iter += 1
            lrate = lr_poly_decay(model, base_lr, curr_iter,
                                  max_iter, power=0.5)
            train_result.append(res)

        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)
        print('Train result {:s}:\n{:s}'.format(nnarch.model.metrics_names, train_result))
        print('\nEvaluating dev set ...')
        result = nnarch.model.evaluate(img_dev, mask_dev, batch_size=32)
        result = np.round(result, decimals=10)
        print('\nDev set result {:s}:\n{:s}'.format(nnarch.model.metrics_names, result))
        save_file = '_'.join(['rvsc', contour_type,
                              'epoch', str(e + 1)]) + '.h5'

        if not os.path.exists('model_logs'):
            os.makedirs('model_logs')

        save_path = os.path.join('model_logs', save_file)
        print('\nSaving model weights to {:s}'.format(save_path))
        nnarch.save_weights(save_path)
