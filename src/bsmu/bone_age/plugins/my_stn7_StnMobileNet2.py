import os, glob, random, time
import numpy as np
import pandas as pd
# from PIL import Image
import cv2
import math
# from skimage.io import imshow_collection
import skimage.io
import skimage.transform

import keras
from keras.utils import Sequence
from keras import backend as K
from keras.backend import tf as ktf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import InputLayer, Dense, Flatten, Lambda, GlobalAveragePooling2D, Input, concatenate, \
    AveragePooling2D, Cropping2D
from keras import optimizers
from keras import losses
from keras.applications import Xception, DenseNet201
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.xception import preprocess_input
import keras.models
from pathlib import Path
from albumentations import (ShiftScaleRotate, HorizontalFlip, Compose, RandomBrightnessContrast, RandomGamma)
import albumentations
import albumentations.augmentations.transforms
from keras import layers

# import my_lr_multiplier
from keras_lr_multiplier import LRMultiplier


# IMAGE_FILE_SIZE = (512, 512)

BATCH_SIZE = 10
MODEL_INPUT_SIZE = (768, 768)
#MODEL_INPUT_SIZE = (512, 512)
MODEL_INPUT_CHANNELS = 3
SAMPLING_SIZE = (464, 464)
# SAMPLING_SIZE = (192, 192)
MODEL_DIR = Path('../Models')
MODEL_NAME = 'Xception_StnMobileNet2_768_464_b10__8.582.h5'  # _LocnetBiggerAveragePoolLinearAct.h5'
MODEL_PATH = MODEL_DIR / MODEL_NAME
LOG_DIR = MODEL_DIR / 'Logs'
LOG_PATH = LOG_DIR / MODEL_NAME

DATA_PATH = Path('../Data')
TRAIN_DATA_CSV_PATH = DATA_PATH / 'train_bones.csv'
VALID_DATA_CSV_PATH = DATA_PATH / 'valid_bones.csv'
TEST_STN_DATA_CSV_PATH = DATA_PATH / 'test_stn_bones.csv'
#SMALL_IMAGES_PATH = DATA_PATH / 'SmallImages192'

#IMAGES_PATH = DATA_PATH / 'SmallImages512'
IMAGES_PATH = Path('C:/MyDiskBackup/Projects/BoneAge/Data/SmallImages768_Mean')

STN_OUTPUT_PATH = Path('../Temp/STN_output')


def print_info(image, prefix: str = ''):
    print(f'{prefix}\t\t{image.shape}\t{image.min()}\t{image.max()}\t{image.dtype}')


def print_title(title: str):
    print(f'\n\t==={title.upper()}===')


def aug(p=0.5):
    return Compose([
        ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, rotate_limit=20, shift_limit=0.2, scale_limit=0.2, p=1), ## interpolation=cv2.INTER_CUBIC
        HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.5),
        # RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1),

        RandomGamma(p=0.5),  # (gamma_limit=(50, 150), p=1)

        albumentations.IAASharpen(p=0.5),

        albumentations.OpticalDistortion(p=0.5),

        albumentations.RandomBrightnessContrast(p=0.2)

    ], p=p)


augmentation = aug(1)


def normalized_image(image):
    image_min = image.min()
    if image_min != 0:
        image = image - image_min
    image_max = image.max()
    if image_max != 0 and image_max != 1:
        image = image / image_max
    return image


def augmentate_image(image):
    augmentation_results = augmentation(image=image)
    augmented_image = augmentation_results['image']

    # Normalize once again image to [0, 1]
    augmented_image = normalized_image(augmented_image)

    return augmented_image


class DataGenerator(Sequence):
    def __init__(self, images_path: Path, data_csv_path: Path, batch_size, is_train):
        self.images_path = images_path
        self.batch_size = batch_size
        self.is_train = is_train

        data_csv = pd.read_csv(str(data_csv_path))
        self.number_of_images = len(data_csv.index)

        self.image_ids = np.zeros(shape=(self.number_of_images, 1), dtype=np.uint32)
        # self.images = np.zeros(shape=(self.number_of_images, *IMAGE_FILE_SIZE), dtype=np.float32)
        self.ages = np.zeros(shape=(self.number_of_images, 1), dtype=np.float32)
        self.males = np.zeros(shape=(self.number_of_images, 1), dtype=np.uint8)

        for index, csv_row in enumerate(data_csv.values):
            image_id, age, male = csv_row
            print(index, 'image_id, age, male', image_id, age, male)

            self.image_ids[index, ...] = image_id
            self.ages[index, ...] = age / 120 - 1  # scale to [-1, 1]
            self.males[index, ...] = male

        self.image_indexes = np.arange(self.number_of_images)
        self.on_epoch_end()

    def __len__(self):
        """Return number of batches per epoch"""
        return math.ceil(self.number_of_images / self.batch_size)

    def __getitem__(self, batch_index):
        """Generate one batch of data"""
        batch_images = np.zeros(shape=(self.batch_size, *MODEL_INPUT_SIZE, MODEL_INPUT_CHANNELS), dtype=np.float32)
        # For ages use -1 instead of zeros, because for black images age should be 0 months
        batch_ages = np.full(shape=(self.batch_size, 1), fill_value=-1, dtype=np.float32)
        batch_males = np.zeros(shape=(self.batch_size, 1), dtype=np.uint8)

        # Generate image indexes of the batch
        batch_image_indexes = self.image_indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        for item_number, batch_image_index in enumerate(batch_image_indexes):
            image_id = self.image_ids[batch_image_index][0]
            age = self.ages[batch_image_index]
            male = self.males[batch_image_index]

            image_path = self.images_path / f'{image_id}.png'
            image = skimage.io.imread(str(image_path))
            image = normalized_image(image)

            if self.is_train:
                augmented_image = augmentate_image(image)
            else:
                augmented_image = image

            augmented_image = augmented_image * 255
            augmented_image = np.stack((augmented_image,) * MODEL_INPUT_CHANNELS, axis=-1)
            batch_images[item_number, ...] = augmented_image

            batch_ages[item_number, ...] = age
            batch_males[item_number, ...] = male

        batch_images = preprocess_input(batch_images)
        return [batch_images, batch_males], batch_ages

    def on_epoch_end(self):
        """Shuffle files after each epoch"""
        if self.is_train:
            np.random.shuffle(self.image_indexes)


def age_mae(y_true, y_pred):
    y_true = (y_true + 1) * 120
    y_pred = (y_pred + 1) * 120
    return losses.mean_absolute_error(y_true, y_pred)


def train_model(model, epochs, lr=1e-4):
    train_generator = DataGenerator(IMAGES_PATH, TRAIN_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=True)
    valid_generator = DataGenerator(IMAGES_PATH, VALID_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=False)

    checkpoint_callback = ModelCheckpoint(filepath=str(MODEL_PATH), monitor='val_age_mae', verbose=1, save_best_only=True)
    # reduce_lr_callback = ReduceLROnPlateau(monitor='val_age_mae', factor=0.5, patience=10, verbose=1, min_lr=1e-6)
    # early_stopping_callback = EarlyStopping(monitor='val_age_mae', patience=10)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_age_mae', factor=0.5, patience=3, verbose=1, min_lr=1e-10)
    early_stopping_callback = EarlyStopping(monitor='val_age_mae', patience=30)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(LOG_PATH), write_graph=False)
    stn_results_callback = StnResultsCallback()

    callbacks = [checkpoint_callback, reduce_lr_callback, early_stopping_callback, tensorboard_callback, stn_results_callback]


    ### LR multiplier
    #multipliers = {'locnet': 0.08}
    # multipliers = {}
    # opt = my_lr_multiplier.LearningRateMultiplier(optimizers.Adam(lr=lr), lr_multipliers=multipliers)
    # opt = LRMultiplier(optimizers.Adam(lr=lr), multipliers)

    model.compile(optimizer=optimizers.Adam(lr=lr), loss=losses.mae, metrics=[age_mae])

    # model.compile(optimizer=optimizers.Adam(lr=lr), loss=losses.mae, metrics=[age_mae])
    model.fit_generator(generator=train_generator,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=valid_generator)


from keras.layers import Activation
from keras.layers import MaxPool2D, AvgPool2D
from keras.layers import Conv2D

import stn.utils
# from stn.layers import BilinearInterpolation
from stn.new.Stn_Oarriaga import BilinearInterpolation


def show_stn_result():
    model = keras.models.load_model(str(MODEL_PATH),
                                    custom_objects={'BilinearInterpolation': BilinearInterpolation},
                                    compile=False)

    img = skimage.io.imread('../Data/SmallImages192/1399.png')
    gender = 0

    # rescale to [0, 1]
    img = img - img.min()
    image_max = img.max()
    if max != 0:
        img = img / image_max

    # rescale to [0, 255]
    img = img * 255

    img = np.stack((img,) * 3, axis=-1)

    # rescale to [-1, 1] for Xception
    # img = preprocess_input(img)

    images = np.zeros((1, *SMALL_INPUT_SIZE, 3), dtype=np.float32)
    images[0, ...] = img

    images = preprocess_input(images)

    sex = np.zeros((1, 1), dtype=np.uint8)
    sex[0, ...] = gender

    input_1 = model.get_layer('input_1').input
    input_2 = model.get_layer('input_2').input
    output_age = model.layers[-1].output

    # input_image = model.input
    output_STN = model.get_layer('stn_interpolation').output
    #STN_function = K.function([input_image], [output_STN])
    STN_function = K.function([input_1, input_2], [output_age, output_STN])

    [age_result, stn_result] = STN_function([images, sex])
    age_result = (age_result[0, ...] + 1) * 120
    print('AGE', age_result)
    stn_result = stn_result[0, ...]
    print('stn result', stn_result.shape, stn_result.min(), stn_result.max(), stn_result.dtype)
    stn_result[stn_result > 1] = 1
    stn_result[stn_result < -1] = -1
    skimage.io.imsave('STN_result.png', stn_result)

    stn_result = stn_result - stn_result.min()
    stn_result = stn_result / stn_result.max()
    skimage.io.imsave('STN_result2.png', (stn_result * 255).astype(np.int))


def test_generator():
    print_title('test generator')

    generator = DataGenerator(IMAGES_PATH, TRAIN_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=True)
    generator_len = len(generator)
    print('generator_len (number of batches per epoch)', generator_len)

    batch = generator.__getitem__(int(generator_len / 2))
    batch_input, batch_ages = batch
    batch_images, batch_males = batch_input[0], batch_input[1]
    print_info(batch_images, 'batch_images')
    print_info(batch_males, 'batch_males')
    print_info(batch_ages, 'batch_ages')

    # Save all images in batches
    for batch_image_index in range(len(batch_images)):
        image = batch_images[batch_image_index]
        male = batch_males[batch_image_index]
        age = batch_ages[batch_image_index]

        print_info(image, 'image')
        print(male, 'male')
        print(age, 'age')

        image = normalized_image(image)

        skimage.io.imsave(str(Path('../Temp/TestGenerator') / f'{batch_image_index}.png'), image)


class StnResultsCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print('StnResultsCallback on_train_begin')

        self.generator = DataGenerator(IMAGES_PATH, TEST_STN_DATA_CSV_PATH, batch_size=BATCH_SIZE, is_train=False)
        self.batch = self.generator.__getitem__(0)
        self.batch_input, self.batch_ages = self.batch
        self.batch_images, self.batch_males = self.batch_input[0], self.batch_input[1]

        self.batch1 = self.generator.__getitem__(1)
        self.batch_input1, self.batch_ages1 = self.batch1
        self.batch_images1, self.batch_males1 = self.batch_input1[0], self.batch_input1[1]

        self.number_of_images_to_save = 4  # len(self.batch_images)

        self.save_batch_images(self.batch_images, 0)
        self.save_batch_images(self.batch_images1, 1)

        input_1 = self.model.get_layer('input_1').input
        input_male = self.model.get_layer('input_male').input
        output_age = self.model.layers[-1].output

        # input_image = model.input
        output_STN = self.model.get_layer('stn_interpolation').output
        # STN_function = K.function([input_image], [output_STN])

        output_final_conv_layer = self.model.get_layer('xception').get_output_at(-1)  # Take last node from xception
        output_pooling_after_final_conv_layer = self.model.get_layer('encoder_pooling').output

        output_locnet_scale = self.model.get_layer('locnet_scale').output
        output_locnet_scale_activation = self.model.get_layer('locnet_scale_activation').output

        output_locnet_translate = self.model.get_layer('locnet_translate').output
        output_locnet_translate_activation = self.model.get_layer('locnet_translate_activation').output

        output_locnet_last_conv = self.model.get_layer('locnet_encoder').get_output_at(-1)
        output_locnet_pooling = self.model.get_layer('locnet_pooling').output

        self.STN_function = K.function(
            [input_1, input_male],
            [output_age, output_STN, output_final_conv_layer, output_pooling_after_final_conv_layer,
             output_locnet_scale, output_locnet_scale_activation, output_locnet_translate, output_locnet_translate_activation,
             output_locnet_last_conv, output_locnet_pooling])

    def save_batch_images(self, batch_images, batch_number):
        for index in range(self.number_of_images_to_save):
            image = batch_images[index]
            image = normalized_image(image)
            skimage.io.imsave(f'../Temp/STN_output/STN_b{batch_number}_{index}_input_norm.png', (image * 255).astype(np.int))

    def on_epoch_end(self, epoch, logs=None):
        print('\n\tStnResultsCallback on_epoch_end', epoch)

        self.stn_results_for_batch(self.batch_input, epoch, 0)
        self.stn_results_for_batch(self.batch_input1, epoch, 1)

    def build_cam(self, conv, pooling):
        # cam = np.zeros(shape=conv.shape[:2], dtype=np.float32)
        cam = np.copy(conv)
        for feature_map_index in range(cam.shape[2]):
            cam[..., feature_map_index] *= pooling[feature_map_index]

        # print_info(cam, 'cam ---0---')
        cam = np.mean(cam, axis=-1)
        # print_info(cam, 'cam ---1--- mean')
        cam = np.maximum(cam, 0)
        # print_info(cam, 'cam ---2--- maximum')
        cam = cam / np.max(cam)
        # print_info(cam, 'cam ---3--- devide to max')
        return cam

    def stn_results_for_batch(self, batch_input, epoch, batch_number):
        print(f'\n\t=============Batch {batch_number} epoch {epoch}')
        [age_results, stn_results, final_conv_results, pooling_after_final_conv_results,
         locnet_scale_results, locnet_scale_activations_results,
         locnet_translate_results, locnet_translate_activations_results,
         locnet_last_conv_results, locnet_pooling_results] = self.STN_function(batch_input)
        print_info(age_results, 'age_results info')
        print_info(stn_results, 'stn_results info')
        print_info(final_conv_results, 'final_conv_results info')
        print_info(pooling_after_final_conv_results, 'pooling_after_final_conv_results info')
        print_info(locnet_scale_results, 'locnet_scale_results')
        print_info(locnet_scale_activations_results, 'locnet_scale_activations_results')
        print_info(locnet_translate_results, 'locnet_translate_results')
        print_info(locnet_translate_activations_results, 'locnet_translate_activations_results')
        print_info(locnet_last_conv_results, 'locnet_last_conv_results')
        print_info(locnet_pooling_results, 'locnet_pooling_results')

        for index in range(self.number_of_images_to_save):
            input_image = batch_input[0][index, ...]

            print(f'\t---{index}---')
            age_result = (age_results[index, ...] + 1) * 120
            print('AGE', age_result)
            stn_result = stn_results[index, ...]
            print('stn result', stn_result.shape, stn_result.min(), stn_result.max(), stn_result.dtype)
            # skimage.io.imsave(f'../Temp/STN_output/STN_result{index}.png', stn_result)
            print('locnet_scale', locnet_scale_results[index, ...])
            print('locnet_scale_activation', locnet_scale_activations_results[index, ...])
            print('locnet_translate', locnet_translate_results[index, ...])
            print('locnet_translate_activation', locnet_translate_activations_results[index, ...])

            normalized_stn_result = normalized_image(stn_result)
            stn_save_file_name = Path(f'STN_b{batch_number}_{index}_result_norm_epoch_{epoch}.png')
            ### skimage.io.imsave(str(STN_OUTPUT_PATH / stn_save_file_name), (normalized_stn_result * 255).astype(np.int))

            # Save activation map
            cam = self.build_cam(final_conv_results[index, ...], pooling_after_final_conv_results[index, ...])
            '''
            final_conv_result = final_conv_results[index, ...]
            pooling_after_final_conv_result = pooling_after_final_conv_results[index, ...]

            cam = np.zeros(shape=final_conv_result.shape[:2], dtype=np.float32)
            for feature_map_index in range(final_conv_result.shape[2]):
                final_conv_result[..., feature_map_index] *= pooling_after_final_conv_result[feature_map_index]

            # print_info(cam, 'cam ---0---')
            cam = np.mean(final_conv_result, axis=-1)
            # print_info(cam, 'cam ---1--- mean')
            cam = np.maximum(cam, 0)
            # print_info(cam, 'cam ---2--- maximum')
            cam = cam / np.max(cam)
            # print_info(cam, 'cam ---3--- devide to max')
            '''
            cam = cv2.resize(cam, SAMPLING_SIZE, interpolation=cv2.INTER_LINEAR)
            ### skimage.io.imsave(str(STN_OUTPUT_PATH / (stn_save_file_name.name + '_cam.png')), cam)
            # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap[np.where(cam < 0.05)] = 0
            img = heatmap * 0.5 + (normalized_stn_result * 255)
            cv2.imwrite(str(STN_OUTPUT_PATH / (stn_save_file_name.name + '_cam_heat.png')), img)

            # Save locnet activation map
            locnet_cam = self.build_cam(locnet_last_conv_results[index, ...], locnet_pooling_results[index, ...])

            locnet_cam = cv2.resize(locnet_cam, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            ### skimage.io.imsave(str(STN_OUTPUT_PATH / (stn_save_file_name.name + 'locnet_cam.png')), locnet_cam)
            locnet_heatmap = cv2.applyColorMap((255 * locnet_cam).astype(np.uint8), cv2.COLORMAP_JET)
            locnet_heatmap[np.where(locnet_cam < 0.05)] = 0
            locnet_img = locnet_heatmap * 0.5 + (normalized_image(input_image) * 255)
            cv2.imwrite(str(STN_OUTPUT_PATH / (stn_save_file_name.name + '_locnet_cam_heat.png')), locnet_img)


def create_model():
    print_title('create model')

    # Add STN
    input_image = Input(shape=(*MODEL_INPUT_SIZE, MODEL_INPUT_CHANNELS))
    #input_image = Input(shape=(None, None, MODEL_INPUT_CHANNELS))
    '''
    locnet = Conv2D(8, (3, 3), padding='same')(input_image)
    locnet = layers.BatchNormalization()(locnet)
    locnet = layers.Activation('relu')(locnet)
    locnet = Conv2D(8, (3, 3), padding='same')(locnet)
    locnet = layers.BatchNormalization()(locnet)
    locnet = layers.Activation('relu')(locnet)
    locnet = AvgPool2D(pool_size=(2, 2))(locnet)
    # d1 = Dropout(0.2)(m1)

    locnet = Conv2D(16, (3, 3), padding='same')(locnet)
    locnet = layers.BatchNormalization()(locnet)
    locnet = layers.Activation('relu')(locnet)
    locnet = Conv2D(16, (3, 3), padding='same')(locnet)
    locnet = layers.BatchNormalization()(locnet)
    locnet = layers.Activation('relu')(locnet)
    locnet = AvgPool2D(pool_size=(2, 2))(locnet)
    # d2 = Dropout(0.2)(m2)

    locnet = Conv2D(32, (3, 3), padding='same')(locnet)
    locnet = layers.BatchNormalization()(locnet)
    locnet = layers.Activation('relu')(locnet)
    locnet = Conv2D(32, (3, 3), padding='same')(locnet)
    locnet = layers.BatchNormalization()(locnet)
    locnet = layers.Activation('relu')(locnet)
    locnet = AvgPool2D(pool_size=(2, 2))(locnet)
    # d3 = Dropout(0.2)(m3)

    locnet = Conv2D(64, (3, 3), padding='same')(locnet)
    locnet = layers.BatchNormalization()(locnet)
    locnet = layers.Activation('relu')(locnet)
    locnet = Conv2D(64, (3, 3), padding='same')(locnet)
    locnet = layers.BatchNormalization()(locnet)
    locnet = layers.Activation('relu')(locnet)
    locnet = AvgPool2D(pool_size=(2, 2))(locnet)
    # d4 = Dropout(0.2)(m4)

    locnet = Conv2D(128, (3, 3), padding='same')(locnet)
    locnet = layers.BatchNormalization()(locnet)
    locnet = layers.Activation('relu')(locnet)
    locnet = Conv2D(128, (3, 3), padding='same')(locnet)
    locnet = layers.BatchNormalization()(locnet)
    locnet = layers.Activation('relu')(locnet)
    locnet = AvgPool2D(pool_size=(2, 2), name='locnet_encoder')(locnet)
    '''
    '''
    locnet = AvgPool2D(pool_size=(4, 4))(input_image)
    locnet_encoder_model = Xception(include_top=False, weights='imagenet', pooling=None)
                                       #input_shape=(*SAMPLING_SIZE, MODEL_INPUT_CHANNELS), pooling=None)
    locnet_encoder_model.name = 'locnet_encoder'
    locnet = locnet_encoder_model(locnet)
    '''

    # locnet = MaxPool2D(pool_size=(2, 2))(input_image)
    # locnet = Conv2D(20, (5, 5))(locnet)
    # # locnet = layers.Activation('relu')(locnet)
    #
    # locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    # locnet = Conv2D(50, (5, 5))(locnet)
    # # locnet = layers.Activation('relu')(locnet)
    #
    # '''
    # locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    # locnet = layers.SeparableConv2D(100, (3, 3), padding='same', use_bias=False)(locnet)
    # locnet = layers.Activation('relu')(locnet)
    #
    # locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    # locnet = layers.SeparableConv2D(200, (3, 3), padding='same', use_bias=False)(locnet)
    # locnet = layers.Activation('relu')(locnet)
    # '''
    #
    # locnet = layers.SeparableConv2D(100, (5, 5), use_bias=False)(locnet)
    # # locnet = layers.Activation('relu')(locnet)
    #
    # locnet = layers.SeparableConv2D(200, (5, 5), use_bias=False, name='locnet_last_conv')(locnet)
    # # locnet = layers.Activation('relu')(locnet)

    '''

    locnet = MaxPool2D(pool_size=(2, 2))(input_image)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = layers.Activation('relu')(locnet)

    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = layers.Activation('relu')(locnet)

    locnet = layers.SeparableConv2D(40, (3, 3), padding='same', use_bias=False)(locnet)
    locnet = layers.Activation('relu')(locnet)

    locnet = layers.SeparableConv2D(40, (3, 3), padding='same', use_bias=False)(locnet)
    locnet = layers.Activation('relu')(locnet)

    locnet = layers.GlobalMaxPooling2D()(locnet)
    '''

    '''
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    '''

    downsample = MaxPool2D(pool_size=(4, 4), name='locnet_downsample')(input_image)

    locnet_encoder_model = MobileNetV2(include_top=False, weights='imagenet', pooling=None, input_shape=(192, 192, MODEL_INPUT_CHANNELS))
    locnet_encoder_model.name = 'locnet_encoder'
    locnet = locnet_encoder_model(downsample)  # encoder_model.output

    # locnet = Flatten()(locnet)
    locnet_pooling = layers.GlobalAveragePooling2D(name='locnet_pooling')(locnet)

    input_male = Input(shape=(1,), name='input_male')

    '''
    locnet_x_male = Dense(32, activation='relu')(input_male)

    locnet_x = concatenate([locnet_pooling, locnet_x_male], axis=-1)
    locnet_x = Dense(1024, activation='relu')(locnet_x)
    locnet_x = Dense(1024, activation='relu')(locnet_x)
    locnet_output = Dense(1, activation='tanh')(locnet_x)
    '''

    # locnet = Dense(200)(locnet)
    locnet = Dense(40, name='locnet_dense_1')(locnet_pooling)
    locnet = layers.BatchNormalization(name='locnet_batch_norm_1')(locnet)
    locnet = Activation('relu', name='locnet_activation_1')(locnet)
    # locnet = Activation('tanh')(locnet)  ### ??

    locnet_scale = Dense(10, name='locnet_before_scale_dense')(locnet)
    locnet_scale = layers.BatchNormalization(name='locnet_before_scale_batch_norm')(locnet_scale)
    locnet_scale = Activation('relu', name='locnet_before_scale_activation')(locnet_scale)
    # locnet_scale = Activation('tanh')(locnet_scale)

    # locnet = Activation('linear')(locnet)  ### ??
    scale_weights = stn.utils.get_initial_weights_for_scale(10)
    locnet_scale = Dense(1, weights=scale_weights, name='locnet_scale')(locnet_scale)
    locnet_scale = layers.BatchNormalization(name='locnet_scale_batch_norm')(locnet_scale)
###    locnet_scale = Activation('sigmoid')(locnet_scale)
    locnet_scale = Activation('sigmoid', name='locnet_scale_activation')(locnet_scale)

    locnet_translate = Dense(15, name='locnet_before_translate_dense')(locnet)
    locnet_translate = layers.BatchNormalization(name='locnet_before_translate_batch_norm')(locnet_translate)
    locnet_translate = Activation('relu', name='locnet_before_translate_activation')(locnet_translate)
    # locnet_translate = Activation('tanh')(locnet_translate)
    translate_weights = stn.utils.get_initial_weights_for_translate(15)
    locnet_translate = Dense(2, weights=translate_weights, name='locnet_translate')(locnet_translate)
    locnet_translate = layers.BatchNormalization(name='locnet_translate_batch_norm')(locnet_translate)
###    locnet_translate = Activation('tanh')(locnet_translate)
    locnet_translate = Activation('tanh', name='locnet_translate_activation')(locnet_translate)

    x = BilinearInterpolation(SAMPLING_SIZE, name='stn_interpolation')([input_image, locnet_scale, locnet_translate])

    encoder_model = Xception(include_top=False, weights='imagenet', pooling=None)
                            # input_shape=(*SAMPLING_SIZE, MODEL_INPUT_CHANNELS), pooling=None)
    # encoder_model.summary()
    # input_image = encoder_model.input
###    input_male = Input(shape=(1,))

    # x_image = encoder_model(input_image)
    x_image = encoder_model(x)  # encoder_model.output
#    x_image = layers.GlobalMaxPooling2D()(x_image)
    x_image = layers.GlobalAveragePooling2D(name='encoder_pooling')(x_image)

    x_male = Dense(32, activation='relu')(input_male)

    x = concatenate([x_image, x_male], axis=-1)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
#   x = Dense(1024, activation='tanh')(x)
#    output = Dense(1, activation='linear')(x)
    output = Dense(1, activation='tanh')(x)

    # output = concatenate([locnet_output, output], axis=-1)
    # output = Dense(1, activation='tanh')(output)

    model = Model(inputs=[input_image, input_male], outputs=output)
    return model


def main():
    print(f'Tensorflow version: {ktf.__version__}')

    # test_generator()
    # exit()

    if MODEL_PATH.exists():
        print_title('load model')
        model = keras.models.load_model(str(MODEL_PATH),
                                        custom_objects={'BilinearInterpolation': BilinearInterpolation},
                                        compile=False)
    else:
        model = create_model()


    for i in range(len(model.layers)):
        print(i, model.layers[i], '      --- ', model.layers[i].name, model.layers[i].trainable)
    model.summary()


    ##################################
    # Set all layers to trainable
    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    stn_interpolation_layer = model.get_layer('stn_interpolation')
    stn_interpolation_layer_index = model.layers.index(stn_interpolation_layer)

    # for i in range(stn_interpolation_layer_index + 1, len(model.layers)):
    #     model.layers[i].trainable = False

    for i in range(stn_interpolation_layer_index + 1):
        model.layers[i].trainable = False

    print('================== After FREZE ===============')
    for i in range(len(model.layers)):
        print(i, model.layers[i], '      --- ', model.layers[i].name, model.layers[i].trainable)

    '''
    stn_interpolation_layer = model.get_layer('stn_interpolation')
    stn_interpolation_layer_index = model.layers.index(stn_interpolation_layer)

    train_stn = False
    loops = 10
    for loop_index in range(loops):
        print(f'\n\n\n\t\t*** LOOP {loop_index}   train_stn: {train_stn} ***\n')

        # Set all layers to trainable
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        if train_stn:
            for i in range(stn_interpolation_layer_index + 1, len(model.layers)):
                model.layers[i].trainable = False
            epochs = 5
            lr = 3e-5
        else:
            for i in range(stn_interpolation_layer_index + 1):
                model.layers[i].trainable = False
            epochs = 15
            lr = 1e-4

        print('================== After FREZE ===============')
        for i in range(len(model.layers)):
            print(i, model.layers[i], '      --- ', model.layers[i].name, model.layers[i].trainable)

        train_model(model, epochs=epochs, lr=lr)

        train_stn = not train_stn
    ##################################
    '''

    train_model(model, epochs=100, lr=8e-6)

    # show_stn_result()


if __name__ == '__main__':
    main()
