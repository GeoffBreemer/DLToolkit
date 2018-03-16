from dltoolkit.nn.segment import FCN32_VGG16_NN
from keras.optimizers import RMSprop
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Cropping2D, \
    Dropout, Activation, Reshape, Conv2DTranspose, merge, Flatten, Dense, Permute, ZeroPadding2D, Convolution2D, Deconvolution2D
from keras.models import Model, Sequential

IMG_HEIGHT = 320
IMG_WIDTH = 320
IMG_CHANNELS = 3
NUM_CLASSES = 2

# def convblock(cdim, nb, bits=3):
# 	L = []
#
# 	for k in range(1, bits + 1):
# 		convname = 'conv' + str(nb) + '_' + str(k)
# 		if False:
# 			# first version I tried
# 			L.append(ZeroPadding2D((1, 1)))
# 			L.append(Convolution2D(cdim, kernel_size=(3, 3), activation='relu', name=convname))
# 		else:
# 			L.append(Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname))
#
# 	L.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# 	return L
#
#
# def fcn32_blank(image_height, image_width):
#     withDO = False  # no effect during evaluation but usefull for fine-tuning
#     mdl = Sequential()
#
#     # First layer is a dummy-permutation = Identity to specify input shape
#     mdl.add(Permute((1, 2, 3), input_shape=(image_height, image_width, IMG_CHANNELS)))  # WARNING : axis 0 is the sample dim
#
#     for l in convblock(64, 1, bits=2):
#         mdl.add(l)
#
#     for l in convblock(128, 2, bits=2):
#         mdl.add(l)
#
#     for l in convblock(256, 3, bits=3):
#         mdl.add(l)
#
#     for l in convblock(512, 4, bits=3):
#         mdl.add(l)
#
#     for l in convblock(512, 5, bits=3):
#         mdl.add(l)
#
#     mdl.add(Convolution2D(4096, kernel_size=(7, 7), padding='same', activation='relu', name='fc6'))  # WARNING border
#     mdl.add(Convolution2D(4096, kernel_size=(1, 1), padding='same', activation='relu', name='fc7'))  # WARNING border
#     mdl.add(Convolution2D(NUM_CLASSES, kernel_size=(1, 1), padding='same', activation='relu', name='score_fr'))
#
#     convsize = mdl.layers[-1].output_shape[2]
#     deconv_output_size = (convsize - 1) * 2 + 4  # INFO: =34 when images are 512x512
#     print("deconv_output_size: {}".format(deconv_output_size))
#     #  print("deconv_output_size: {}".format(deconv_output_size))
#     #  WARNING : valid, same or full ?
#     mdl.add(Deconvolution2D(NUM_CLASSES, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation=None, name='score2'))
#
#     extra_margin = deconv_output_size - convsize * 2  # INFO: =2 when images are 512x512
#     print("extra margin: {}".format(extra_margin))
#     assert (extra_margin > 0)
#     assert (extra_margin % 2 == 0)
#     # INFO : cropping as deconv gained pixels
#     # print(extra_margin)
#     c = ((0, extra_margin), (0, extra_margin))
#     # print(c)
#     mdl.add(Cropping2D(cropping=c))
#     # print(mdl.summary())
#
#     return mdl
#
#
# def FCN32(n_classes, input_height=416, input_width=608, vgg_level=3):
#
#     assert input_height % 32 == 0
#     assert input_width % 32 == 0
#
#     # https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
#     img_input = Input(shape=(input_height, input_width, 3))
#
#     x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
#         img_input)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
#     f1 = x
#     # Block 2
#     x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
#     x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
#     f2 = x
#
#     # Block 3
#     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
#     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
#     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#     f3 = x
#
#     # Block 4
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
#     f4 = x
#
#     # Block 5
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
#     f5 = x
#
#     x = Flatten(name='flatten')(x)
#     x = Dense(4096, activation='relu', name='fc1')(x)
#     x = Dense(4096, activation='relu', name='fc2')(x)
#     x = Dense(1000, activation='softmax', name='predictions')(x)
#
#     vgg = Model(img_input, x)
#     # vgg.load_weights(VGG_Weights_path)
#
#     o = f5
#
#     o = (Conv2D(4096, (7, 7), activation='relu', padding='same'))(o)
#     # o = Dropout(0.5)(o)
#     o = (Conv2D(4096, (1, 1), activation='relu', padding='same'))(o)
#     # o = Dropout(0.5)(o)
#
#     o = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal'))(o)
#     o = Conv2DTranspose(filters=n_classes, kernel_size=(64, 64), strides=(32, 32), padding="valid", use_bias=False)(
#         o)
#     o_shape = Model(img_input, o).output_shape
#
#     outputHeight = o_shape[1]
#     outputWidth = o_shape[2]
#
#     print(o_shape)
#
#     o = (Reshape((outputHeight * outputWidth, -1)))(o)
#     o = (Activation('softmax'))(o)
#     model = Model(img_input, o)
#     model.outputWidth = outputWidth
#     model.outputHeight = outputHeight
#
#     return model


if __name__ == "__main__":
    fcn = FCN32_VGG16_NN(img_height=IMG_HEIGHT,
                         img_width=IMG_WIDTH,
                         img_channels=IMG_CHANNELS,
                         num_classes=NUM_CLASSES)


    model, output_shape = fcn.build_model()
    model.summary()
    print("output shape: {}".format(output_shape))

    fcn.freeze_vgg16()

    # opt = RMSProp(lr=0.001)

    # print("11111111111111")
    # model = FCN32(NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH)
    # model.summary()

    # print("22222222222222")
    # model = fcn32_blank(IMG_HEIGHT, IMG_WIDTH)
    # model.summary()

    # for (i, layer) in enumerate(model.layers[1].layers):
    #     print("[INFO] {}\t{}".format(i, layer.__class__.__name__))

