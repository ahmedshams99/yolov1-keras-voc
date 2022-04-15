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
    x = Conv2D(16, (3, 3), padding='same', name='convolutional_0',
               kernel_regularizer=l2(5e-4))(inputs)
    x = BatchNormalization(name='bnconvolutional_0')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same', name='convolutional_1',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same', name='convolutional_2',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same', name='convolutional_3',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), padding='same', name='convolutional_4',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(512, (3, 3), padding='same', name='convolutional_5',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(1024, (3, 3), padding='same', name='convolutional_6',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (3, 3), padding='same', name='convolutional_7',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Flatten()(x)
    x = Dense(1470, activation='linear', name='connected_0')(x)
    # outputs = Reshape((7, 7, 30))(x)
    outputs = Yolo_Reshape((7, 7, 30))(x)

    return outputs
def model_yolov1(inputs):
    x = Conv2D(filters=64, kernel_size= (7, 7), padding='same', name='convolutional_0',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_0')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    
    x = Conv2D(filters=192, kernel_size= (3, 3), padding='same', name='convolutional_1',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    

    x = Conv2D(filters=128, kernel_size= (1, 1), padding='same', name='convolutional_2',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=256, kernel_size= (3, 3), padding='same', name='convolutional_3',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=256, kernel_size= (1, 1), padding='same', name='convolutional_4',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (3, 3), padding='same', name='convolutional_5',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    
    x = Conv2D(filters=256, kernel_size= (1, 1), padding='same', name='convolutional_6',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_6')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (3, 3), padding='same', name='convolutional_7',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_7')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=256, kernel_size= (1, 1), padding='same', name='convolutional_8',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (3, 3), padding='same', name='convolutional_9',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_9')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=256, kernel_size= (1, 1), padding='same', name='convolutional_10',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_10')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (3, 3), padding='same', name='convolutional_11',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_11')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=256, kernel_size= (1, 1), padding='same', name='convolutional_12',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_12')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (3, 3), padding='same', name='convolutional_13',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_13')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (1, 1), padding='same', name='convolutional_14',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_14')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_15',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_15')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    
    
    x = Conv2D(filters=512, kernel_size= (1, 1), padding='same', name='convolutional_16',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_16')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_17',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_17')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=512, kernel_size= (1, 1), padding='same', name='convolutional_18',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_18')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_19',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_19')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_20',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_20')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', strides=(2, 2), name='convolutional_21',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_21')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_22',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_22')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters=1024, kernel_size= (3, 3), padding='same', name='convolutional_23',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_23')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid')(x)
    x = Dropout(0.5)(x)
    x = Dense(1225, activation='sigmoid')(x)
    outputs = Yolo_Reshape((7, 7, 25))(x)
    return outputs
