from keras.layers import Conv2D, MaxPooling2D, \
    Flatten, Dense, Reshape, LeakyReLU, BatchNormalization, Layer
from keras.regularizers import l2
import keras.backend as K


class Yolo_Reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        super(Yolo_Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def call(self, inputs, **kwargs):
        S = [self.target_shape[0], self.target_shape[1]]
        C = 20
        B = 2
        idx1 = S[0] * S[1] * C
        idx2 = idx1 + S[0] * S[1] * B
        # class prediction
        class_probs = K.reshape(
            inputs[:, :idx1], (K.shape(inputs)[0],) + tuple([S[0], S[1], C]))
        class_probs = K.softmax(class_probs)
        # confidence
        confs = K.reshape(
            inputs[:, idx1:idx2], (K.shape(inputs)[0],) + tuple([S[0], S[1], B]))
        confs = K.sigmoid(confs)
        # boxes
        boxes = K.reshape(
            inputs[:, idx2:], (K.shape(inputs)[0],) + tuple([S[0], S[1], B * 4]))
        boxes = K.sigmoid(boxes)
        # return np.array([class_probs, confs, boxes])
        outputs = K.concatenate([class_probs, confs, boxes])
        return outputs


def model_tiny_yolov1(inputs):
    x = Conv2D(16, (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same', name='convolutional_1', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_1', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same', name='convolutional_2', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_2', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same', name='convolutional_3', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_3', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), padding='same', name='convolutional_4', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_4', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(512, (3, 3), padding='same', name='convolutional_5', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_5', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(1024, (3, 3), padding='same', name='convolutional_6', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_6', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (3, 3), padding='same', name='convolutional_7', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_7', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Flatten()(x)
    x = Dense(1470, activation='linear', name='connected_0')(x)
    # outputs = Reshape((7, 7, 30))(x)
    outputs = Yolo_Reshape((7, 7, 30))(x)

    return outputs
def model_yolov1(inputs):
    x = Conv2D(filters=64, kernel_size= (7, 7), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    
    x = Conv2D(filters=192, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    

    x = Conv2D(filters=128, kernel_size= (1, 1), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=256, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=256, kernel_size= (1, 1), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    
    x = Conv2D(filters=256, kernel_size= (1, 1), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=256, kernel_size= (1, 1), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=256, kernel_size= (1, 1), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=256, kernel_size= (1, 1), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (1, 1), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    
    
    x = Conv2D(filters=512, kernel_size= (1, 1), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (1, 1), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', strides=(2, 2), name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = LocallyConnected2D(filters = filters, kernel_size=size, strides=stride, padding=padding, implementation=2)(x)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1470, activation='sigmoid')(x)
    outputs = Yolo_Reshape((7, 7, 30))(x)
    return outputs
