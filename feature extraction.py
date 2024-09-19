if True:
    from reset_random import reset_random

    reset_random()
import glob
import os
import time

import cmapy
import cv2
import numpy as np
import tqdm
from keras.applications.nasnet import NASNetLarge, preprocess_input
from keras.models import Model
from keras_preprocessing.image import load_img, img_to_array

from utils import CLASSES

SHAPE = (224, 224, 3)


def get_nasnet_large_model():
    print("[INFO] Building NasNetLarge Model")
    model = NASNetLarge(weights="imagenet", include_top=False, input_shape=SHAPE)
    return model


def get_feature_map_model(model):
    layer_outputs = [layer.output for layer in model.layers[1:]]
    feature_map_model = Model(model.input, layer_outputs)
    return feature_map_model


def get_image_to_predict(im_path):
    img = load_img(im_path, target_size=SHAPE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def get_feature(img, model):
    feature = model.predict(img, verbose=False)[0][0][0]
    feat = feature.flatten()
    return feat.tolist()


def get_feature_image(img, model):
    feature_map = model.predict(img, verbose=False)[5]
    feature_image = feature_map[0, :, :, -1]
    feature_image -= feature_image.mean()
    feature_image /= feature_image.std()
    feature_image *= 64
    feature_image += 128
    feature_image = np.clip(feature_image, 0, 255).astype("uint8")
    return feature_image


if __name__ == "__main__":
    DATA_DIR = "Data/preprocessed"
    SAVE_DIR = "Data/features"

    nnl = get_nasnet_large_model()

    nnl_fmm = get_feature_map_model(nnl)

    features = []
    labels = []
    for cls in CLASSES:
        sd = os.path.join(SAVE_DIR, cls)
        os.makedirs(sd, exist_ok=True)
        images_list = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.jpg")))
        for img_path in tqdm.tqdm(
                images_list, desc="[INFO] Extracting Features For Class :: {0}".format(cls)
        ):
            im_name = os.path.basename(img_path)
            im_save_path = os.path.join(sd, im_name)
            nnl_im = get_image_to_predict(img_path)
            nnl_fe = get_feature(nnl_im, nnl)
            features.append(nnl_fe)
            labels.append(CLASSES.index(cls))
            nnl_fm = get_feature_image(nnl_im, nnl_fmm)
            nnl_fm = cv2.resize(nnl_fm, SHAPE[:-1])
            nnl_fm = cv2.applyColorMap(nnl_fm, cmapy.cmap("viridis"))
            cv2.imwrite(im_save_path, nnl_fm)
        time.sleep(0.1)
    features = np.array(features, ndmin=2)
    print('[INFO] Features Shape :: {0}'.format(features.shape))
    print("[INFO] Saving Features and Labels")
    f_path = os.path.join(SAVE_DIR, "features.npy")
    np.save(f_path, features)
    labels = np.array(labels)
    l_path = os.path.join(SAVE_DIR, "labels.npy")
    np.save(l_path, labels)
