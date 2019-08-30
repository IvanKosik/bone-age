"""
Initial version of these code was taken from this file
https://github.com/oarriaga/STN.keras/blob/master/src/models/layers.py

1) Was changed some methods, to use tensorflow intead of keras.backend to improve performance on big images
These file contains tensorflow version of some operations:
https://github.com/tensorflow/models/blob/master/research/transformer/spatial_transformer.py

2) And changed affine transformation matrix to use only scale and translate
"""


from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

if K.backend() == 'tensorflow':
    import tensorflow as tf

    def K_meshgrid(x, y):
        return tf.meshgrid(x, y)

    def K_linspace(start, stop, num):
        return tf.linspace(start, stop, num)

else:
    raise Exception("Only 'tensorflow' is supported as backend")


class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['output_size'] = self.output_size
        return config

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation_scale, transformation_translate = tensors
        output = self._transform(X, transformation_scale, transformation_translate, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, output_size):

        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')

        x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * K.cast(height, dtype='float32')

        print('_interpolate x,y')

        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)

        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        print('_interpolate clip')



        xxx = tf.range(batch_size) * (height * width)
        n_repeats = output_size[0] * output_size[1]

        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        xxx = tf.matmul(tf.reshape(xxx, (-1, 1)), rep)
        tf_result = tf.reshape(xxx, [-1])

        print('tf_shape', tf.shape(tf_result))
        base = tf_result
        # exit()

        '''
        pixels_batch = K.arange(0, batch_size) * (height * width)
        pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        print('_interpolate expand_dims')
        print(f'pixels_batch {pixels_batch}  flat_output_size {flat_output_size}')
        base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        print('_interpolate repeat_elements')
        base = K.flatten(base)
        print('keras_shape', K.shape(base))
        '''

        print('_interpolate repeat_elements,flatten')

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.reshape(image, shape=(-1, num_channels))
        flat_image = K.cast(flat_image, dtype='float32')
        pixel_values_a = K.gather(flat_image, indices_a)
        pixel_values_b = K.gather(flat_image, indices_b)
        pixel_values_c = K.gather(flat_image, indices_c)
        pixel_values_d = K.gather(flat_image, indices_d)

        x0 = K.cast(x0, 'float32')
        x1 = K.cast(x1, 'float32')
        y0 = K.cast(y0, 'float32')
        y1 = K.cast(y1, 'float32')

        area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = K.flatten(grid)
###        grids = K.tile(grid, K.stack([batch_size]))
        grids = tf.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, affine_transformation_scale, affine_transformation_translate, output_size):
        batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]

        # print('af shape', affine_transformation.shape)
        # z = tf.zeros(shape=(tf.shape(affine_transformation)[0], 1), dtype=tf.dtypes.float32)
        z = tf.zeros(shape=(batch_size, 1), dtype='float32')
        transform_tensor = tf.stack([affine_transformation_scale[:, 0], z[:, 0], affine_transformation_translate[:, 0],
                                     z[:, 0], affine_transformation_scale[:, 0], affine_transformation_translate[:, 1]],
                                    axis=1)
        '''
        print('batch_size', tf.Session().run(batch_size))
        print('affine_TRANS', K.shape(affine_transformation))
        transform_matrix_numpy = np.zeros(shape=(batch_size, 6), dtype=np.float32)
        for batch_index in range(batch_size):
            affine_transformation_batch = affine_transformation[batch_index]
            transform_matrix_numpy[batch_index] = [affine_transformation_batch[0], 0, affine_transformation_batch[1],
                                                   0, affine_transformation_batch[0], affine_transformation_batch[2]]
        # transform_matrix = [affine_transformation[0], 0, affine_transformation[1],
        #                     0, affine_transformation[0], affine_transformation[2]]
        transform_tensor = tf.convert_to_tensor(transform_matrix_numpy, np.float32)
        '''
        transformations = K.reshape(transform_tensor,
                                    shape=(batch_size, 2, 3))
        # Replace sentences above with one below to get all transformation operations
        # transformations = K.reshape(affine_transformation,
        #                             shape=(batch_size, 2, 3))


        ###% transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        print('self._make_regular_grids')
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = K.batch_dot(transformations, regular_grids)
        print('self._interpolate')
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        print('after self._interpolate')
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = K.reshape(interpolated_image, new_shape)
        print('return interpolated_image')
        return interpolated_image
