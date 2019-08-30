from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import skimage.transform
import skimage.io

from PySide2.QtCore import QObject, Signal

import keras

from bsmu.vision_core.image import FlatImage
from bsmu.vision.app.plugin import Plugin

# import bsmu.bone_age.plugins.Stn_Oarriaga
from bsmu.bone_age.plugins import stn_layer


class BoneAgeAnalyzerPlugin(Plugin):
    def __init__(self, app: App):
        super().__init__(app)

        self.data_storage = app.enable_plugin('bsmu.vision.plugins.storages.data.DataStoragePlugin').data_storage

        self.united_config = self.config()
        stn_model_path = (Path(__file__).parent / self.united_config.data['stn_model_path']).resolve()
        self.bone_age_analyzer = BoneAgeAnalyzer(stn_model_path)

    def _enable(self):
        self.data_storage.data_added.connect(self.bone_age_analyzer.analyze_bone_age)
        # TODO: add action on self.data_storage.data_removed

    def _disable(self):
        self.data_storage.data_added.disconnect(self.bone_age_analyzer.analyze_bone_age)


class BoneAgeAnalyzer(QObject):
    bone_age_analyzed = Signal()

    def __init__(self, stn_model_path: Path):
        super().__init__()

        self.stn_model_path = stn_model_path
        print('STN model path:', self.stn_model_path.absolute().resolve())

        self._stn_model = None
        self._stn_model_output_function = None

    def analyze_bone_age(self, data: Data):
        if not isinstance(data, FlatImage):
            return

        if self._stn_model is None:
            assert self.stn_model_path.exists(), 'There is no STN model file'

            self._stn_model = keras.models.load_model(str(self.stn_model_path),
                                                      custom_objects={'BilinearInterpolation': stn_layer.BilinearInterpolation},
                                                      compile=False)
            print('MODEL LOADED')

            image_input_layer = self._stn_model.get_layer('input_1').input
            male_input_layer = self._stn_model.get_layer('input_male').input
            age_layer_output = self._stn_model.layers[-1].output
            stn_layer_output = self._stn_model.get_layer('stn_interpolation').output
            final_conv_layer_output = self._stn_model.get_layer('xception').get_output_at(
                -1)  # Take last node from layer
            final_pooling_layer_output = self._stn_model.get_layer('encoder_pooling').output

            self._stn_model_output_function = keras.backend.function(
                [image_input_layer, male_input_layer],
                [age_layer_output, stn_layer_output, final_conv_layer_output, final_pooling_layer_output])

        image = data.array
        print('image_info:', image.shape, image.min(), image.max(), image.dtype)

        BATCH_SIZE = 1
        INPUT_SIZE = (768, 768)
        INPUT_CHANNELS = 3

        input_batch_images = np.zeros(shape=(BATCH_SIZE, *INPUT_SIZE, INPUT_CHANNELS), dtype=np.float32)
        input_batch_images[0, ...] = preprocessed_image(image, INPUT_SIZE, INPUT_CHANNELS)
        input_batch_images = keras.applications.xception.preprocess_input(input_batch_images)
        print('preprocessed image', input_batch_images[0].shape, input_batch_images[0].min(),
              input_batch_images[0].max(), input_batch_images[0].dtype, np.mean(input_batch_images[0]))
        skimage.io.imsave('test_image2.png', input_batch_images[0])

        input_batch_males = np.zeros(shape=(BATCH_SIZE, 1), dtype=np.uint8)
        input_batch_males[0, ...] = 1  # We do not know the gender at first, so use male

        input_batch = [input_batch_images, input_batch_males]
        [age_output_batch, stn_output_batch, final_conv_output_batch, final_pooling_output_batch] = \
            self._stn_model_output_function(input_batch)

        age_in_months = (age_output_batch[0] + 1) * 120
        age_in_years_fractional_part, age_in_years_integer_part = math.modf(age_in_months / 12)
        age_str = f'{age_in_years_integer_part} лет {age_in_years_fractional_part * 12} месяцев'

        print('Age_output', data.path, age_output_batch[0], '\t\t', age_str)
        skimage.io.imsave('test_stn.png', stn_output_batch[0])

        self.bone_age_analyzed.emit()


def preprocessed_image(image, size, channels):
    image = add_pads(image)[0]
    image = skimage.transform.resize(image, size, anti_aliasing=True, order=1).astype(np.float32)

    image = normalized_image(image)
    image = image * 255  # xception.preprocess_input needs RGB values within [0, 255].

    # augmented_image = augmented_image * 255

    if image.ndim == 2:
        image = np.stack((image,) * channels, axis=-1)
    # augmented_image = np.stack((augmented_image,) * MODEL_INPUT_CHANNELS, axis=-1)

    skimage.io.imsave('test_image.png', image)

    return image


def add_pads(image, dims=2):
    """Add zero-padding to make square image (original image will be in the center of paddings)"""
    image_shape = image.shape[:dims]
    pads = np.array(image_shape).max() - image_shape
    print('pads', pads)
    # Add no pads for channel axis
    pads = np.append(pads, [0] * (len(image.shape) - dims)).astype(pads.dtype)
    print('pads', pads)

    # pads[pads < 0] = 0
    before_pads = np.ceil(pads / 2).astype(np.int)
    after_pads = pads - before_pads
    pads = tuple(zip(before_pads, after_pads))
    image = np.pad(image, pads, mode='constant', constant_values=np.mean(image))
    return image, pads


def normalized_image(image):
    image_min = image.min()
    if image_min != 0:
        image = image - image_min
    image_max = image.max()
    if image_max != 0 and image_max != 1:
        image = image / image_max
    return image
