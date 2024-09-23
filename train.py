import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, LeakyReLU, UpSampling2D, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

grayscale_dir = 'data_16/s1'
color_dir = 'data_16/s2'

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
COLOR_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 100

def load_images(image_dir, img_height, img_width, channels):
    images = []
    for img_name in sorted(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        img = imread(img_path)
        img = resize(img, (img_height, img_width, channels), mode='constant', preserve_range=True)
        images.append(img)
    images = np.array(images).astype('float32') / 127.5 - 1
    return images

grayscale_images = load_images(grayscale_dir, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
color_images = load_images(color_dir, IMG_HEIGHT, IMG_WIDTH, COLOR_CHANNELS)

X_train, X_val, y_train, y_val = train_test_split(grayscale_images, color_images, test_size=0.1, random_state=42)

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20)
train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
validation_generator = datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

class SpectralNormalization(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(**kwargs)
        self.layer = layer

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=self.layer.kernel.shape,
            initializer='glorot_uniform',
            trainable=True
        )
        super(SpectralNormalization, self).build(input_shape)

    def call(self, inputs):
        kernel = self.kernel
        kernel = tf.nn.l2_normalize(kernel, axis=None)
        outputs = tf.nn.conv2d(inputs, kernel, strides=self.layer.strides, padding=self.layer.padding.upper())
        return outputs

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = SpectralNormalization(Conv2D(filters, kernel_size, strides=stride, padding='same'))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpectralNormalization(Conv2D(filters, kernel_size, strides=stride, padding='same'))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    return ReLU()(x)

def self_attention(x):
    channels = x.shape[-1]
    f = SpectralNormalization(Conv2D(channels // 8, kernel_size=1))(x)
    g = SpectralNormalization(Conv2D(channels // 8, kernel_size=1))(x)
    h = SpectralNormalization(Conv2D(channels, kernel_size=1))(x)

    f = tf.keras.layers.Reshape((-1, channels // 8))(f)
    g = tf.keras.layers.Reshape((-1, channels // 8))(g)
    h = tf.keras.layers.Reshape((-1, channels))(h)

    s = tf.matmul(f, g, transpose_b=True)
    beta = tf.nn.softmax(s)

    o = tf.matmul(beta, h)
    o = tf.keras.layers.Reshape(x.shape[1:])(o)

    return Add()([x, o])

def fpn_block(low_res_input, high_res_input, filters):
    high_res_input = UpSampling2D()(high_res_input)
    fusion = Add()([low_res_input, high_res_input])
    fusion = Conv2D(filters, kernel_size=3, padding='same')(fusion)
    fusion = BatchNormalization()(fusion)
    return ReLU()(fusion)

def build_generator_with_fpn_and_attention(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    e1 = SpectralNormalization(Conv2D(64, 4, strides=2, padding='same'))(inputs)
    e1 = LeakyReLU(0.2)(e1)

    e2 = SpectralNormalization(Conv2D(128, 4, strides=2, padding='same'))(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(0.2)(e2)

    e3 = SpectralNormalization(Conv2D(256, 4, strides=2, padding='same'))(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(0.2)(e3)

    e4 = SpectralNormalization(Conv2D(512, 4, strides=2, padding='same'))(e3)
    e4 = BatchNormalization()(e4)
    e4 = LeakyReLU(0.2)(e4)

    b = SpectralNormalization(Conv2D(1024, 4, strides=2, padding='same'))(e4)
    b = BatchNormalization()(b)
    b = LeakyReLU(0.2)(b)

    for _ in range(3):
        b = residual_block(b, 1024)

    d1 = fpn_block(b, e4, 512)
    d1 = self_attention(d1)

    d2 = fpn_block(d1, e3, 256)
    d2 = self_attention(d2)

    d3 = fpn_block(d2, e2, 128)
    d3 = self_attention(d3)

    d4 = fpn_block(d3, e1, 64)

    outputs = Conv2DTranspose(COLOR_CHANNELS, kernel_size=4, strides=2, padding='same', activation='tanh')(d4)
    return Model(inputs, outputs, name="generator")

def build_discriminator_with_spectral_norm(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    x = SpectralNormalization(Conv2D(64, 4, strides=2, padding='same'))(inputs)
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv2D(128, 4, strides=2, padding='same'))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv2D(256, 4, strides=2, padding='same'))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv2D(512, 4, strides=2, padding='same'))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv2D(1024, 4, strides=2, padding='same'))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1, 4, padding='same')(x)
    return Model(inputs, x, name="discriminator")

def build_vgg19_model(input_shape=(256, 256, 3)):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    vgg.trainable = False
    model = Model(vgg.input, vgg.get_layer('block5_conv4').output)
    return model

def perceptual_loss(y_true, y_pred):
    vgg = build_vgg19_model()
    y_true_vgg = vgg(y_true)
    y_pred_vgg = vgg(y_pred)
    return tf.reduce_mean(tf.abs(y_true_vgg - y_pred_vgg))

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    generator = build_generator_with_fpn_and_attention()
    discriminator = build_discriminator_with_spectral_norm()

    discriminator.compile(optimizer=Adam(2e-4, beta_1=0.5), loss='mse', metrics=['accuracy'])

    input_grayscale = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    generated_color = generator(input_grayscale)
    discriminator.trainable = False
    validity = discriminator(generated_color)

    gan = Model(input_grayscale, [generated_color, validity])
    gan.compile(optimizer=Adam(2e-4, beta_1=0.5), loss=[perceptual_loss, 'binary_crossentropy'], loss_weights=[100, 1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('colorization_model_best.keras', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

history = gan.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS, callbacks=[early_stopping, checkpoint, reduce_lr])

def preprocess_test_image(image_path, img_height=256, img_width=256, channels=1):
    img = imread(image_path)
    img = resize(img, (img_height, img_width, channels), mode='constant', preserve_range=True)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 127.5 - 1
    return img

def postprocess_and_display_result(grayscale_img, colorized_img):
    grayscale_img = np.squeeze(grayscale_img) * 127.5 + 127.5
    colorized_img = np.squeeze(colorized_img) * 127.5 + 127.5
    colorized_img = np.clip(colorized_img, 0, 255).astype(np.uint8)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale_img, cmap='gray')
    plt.title('Grayscale Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(colorized_img)
    plt.title('Colorized Image')
    plt.show()

test_image_path = 'test_sar_image.png'
test_image = preprocess_test_image(test_image_path)
predicted_color_image = generator.predict(test_image)
postprocess_and_display_result(test_image, predicted_color_image)
