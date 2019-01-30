import os
import cv2
from cytomine.models import AttachedFileCollection, Property

from keras.models import *
from keras.layers import *
import numpy as np


def load_model(job, download_path, model_filename="weights.hf5"):
    attached_files = AttachedFileCollection(job).fetch()
    if not (0 < len(attached_files) < 2):
        raise ValueError("More or less than 1 file attached to the Job (found {} file(s)).".format(len(attached_files)))
    attached_file = attached_files[0]
    if attached_file.filename != model_filename:
        raise ValueError(
            "Expected model file name is '{}' (found: '{}').".format(model_filename, attached_file.filename))
    model_path = os.path.join(download_path, model_filename)
    attached_file.download(model_path)
    return model_path


def load_property(job, property_name):
    property = Property(job, key=property_name).fetch()
    return property.value


def one_hot_encode(idx, n, axis=2):
    out = np.zeros(idx.shape + (n,), dtype=int)
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    out[tuple(grid)] = 1
    return out


def load_data(cj, dims, path, is_masks=False, n_classes=2, dtype=np.float32, **monitor_params):
    images = sorted(os.listdir(path))  # to make sure that several calls return the same list
    imgs = np.ndarray([len(images), dims[0], dims[1], dims[2]], dtype=dtype)
    for i, image_name in cj.monitor(enumerate(images), **monitor_params):
        img = cv2.imread(os.path.join(path, image_name))
        resize_dims = dims[1], dims[0]
        if is_masks:
            if img.ndim == 3:  # check if several class channels
                img = img[:, :, 0]
            img = one_hot_encode(img, n=n_classes, axis=2)
            out = np.zeros(dims, img.dtype)
            # resize class mask by nearest interpolation (doesn't make sense to interpolate classes)
            for c in range(n_classes):
                out[:, :, c] = cv2.resize(img[:, :, c], resize_dims, interpolation=cv2.INTER_NEAREST)
        else:
            out = cv2.resize(img, resize_dims, interpolation=cv2.INTER_LINEAR)
        imgs[i, :, :, :] = out
    return imgs


def create_unet(dims=(574, 574, 3), n_classes=2):
    inputs = Input(dims)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(n_classes, 3, activation='softmax', padding='same')(conv9)

    model = Model(inputs=inputs, outputs=[conv9])

    return model
