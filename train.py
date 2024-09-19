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

# Directories (replace with your paths)
grayscale_dir = 'data_16/s1'
color_dir = 'data_16/s2'

# Image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1  # Grayscale input has 1 channel
COLOR_CHANNELS = 3  # Colorized output has 3 channels (RGB)
BATCH_SIZE = 16
EPOCHS = 100

# Load images and normalize to [-1, 1] for tanh activation
def load_images(image_dir, img_height, img_width, channels):
    images = []
    for img_name in sorted(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        img = imread(img_path)
        img = resize(img, (img_height, img_width, channels), mode='constant', preserve_range=True)
        images.append(img)
    images = np.array(images).astype('float32') / 127.5 - 1  # Normalize to [-1, 1]
    return images

# Load grayscale and color images
grayscale_images = load_images(grayscale_dir, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
color_images = load_images(color_dir, IMG_HEIGHT, IMG_WIDTH, COLOR_CHANNELS)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(grayscale_images, color_images, test_size=0.1, random_state=42)

# Data generators
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20)
train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
validation_generator = datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# Spectral Normalization Layer
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

# Residual Block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = SpectralNormalization(Conv2D(filters, kernel_size, strides=stride, padding='same'))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpectralNormalization(Conv2D(filters, kernel_size, strides=stride, padding='same'))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    return ReLU()(x)

# Attention Mechanism (Self-Attention)
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

# Feature Pyramid Block
def fpn_block(low_res_input, high_res_input, filters):
    high_res_input = UpSampling2D()(high_res_input)
    fusion = Add()([low_res_input, high_res_input])
    fusion = Conv2D(filters, kernel_size=3, padding='same')(fusion)
    fusion = BatchNormalization()(fusion)
    return ReLU()(fusion)

# U-Net Generator with FPN, Residual Blocks, and Attention
def build_generator_with_fpn_and_attention(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Encoder (Down-sampling)
    e1 = SpectralNormalization(Conv2D(64, 4, strides=2, padding='same'))(inputs)  # 128x128x64
    e1 = LeakyReLU(0.2)(e1)

    e2 = SpectralNormalization(Conv2D(128, 4, strides=2, padding='same'))(e1)  # 64x64x128
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(0.2)(e2)

    e3 = SpectralNormalization(Conv2D(256, 4, strides=2, padding='same'))(e2)  # 32x32x256
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(0.2)(e3)

    e4 = SpectralNormalization(Conv2D(512, 4, strides=2, padding='same'))(e3)  # 16x16x512
    e4 = BatchNormalization()(e4)
    e4 = LeakyReLU(0.2)(e4)

    # Bottleneck
    b = SpectralNormalization(Conv2D(1024, 4, strides=2, padding='same'))(e4)  # 8x8x1024
    b = BatchNormalization()(b)
    b = LeakyReLU(0.2)(b)

    # Residual blocks in the bottleneck
    for _ in range(3):
        b = residual_block(b, 1024)

    # Decoder (Up-sampling with FPN and Attention)
    d1 = fpn_block(b, e4, 512)  # 16x16x512
    d1 = self_attention(d1)

    d2 = fpn_block(d1, e3, 256)  # 32x32x256
    d2 = self_attention(d2)

    d3 = fpn_block(d2, e2, 128)  # 64x64x128
    d3 = self_attention(d3)

    d4 = fpn_block(d3, e1, 64)  # 128x128x64
   
    outputs = Conv2DTranspose(COLOR_CHANNELS, kernel_size=4, strides=2, padding='same', activation='tanh')(d4)  # 256x256x3
    return Model(inputs, outputs, name="generator")

# Discriminator with Spectral Normalization
def build_discriminator_with_spectral_norm(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    x = SpectralNormalization(Conv2D(64, 4, strides=2, padding='same'))(inputs)  # 128x128x64
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv2D(128, 4, strides=2, padding='same'))(x)  # 64x64x128
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv2D(256, 4, strides=2, padding='same'))(x)  # 32x32x256
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv2D(512, 4, strides=2, padding='same'))(x)  # 16x16x512
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = SpectralNormalization(Conv2D(1024, 4, strides=2, padding='same'))(x)  # 8x8x1024
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1, 4, padding='same')(x)  # 8x8x1
    return Model(inputs, x, name="discriminator")

# VGG19 model for perceptual loss
def build_vgg19_model(input_shape=(256, 256, 3)):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    vgg.trainable = False
    model = Model(vgg.input, vgg.get_layer('block5_conv4').output)
    return model

# Perceptual Loss Function
def perceptual_loss(y_true, y_pred):
    vgg = build_vgg19_model()
    y_true_vgg = vgg(y_true)
    y_pred_vgg = vgg(y_pred)
    return tf.reduce_mean(tf.abs(y_true_vgg - y_pred_vgg))

# Set up MirroredStrategy for distributed training
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Build and compile the models
    generator = build_generator_with_fpn_and_attention()
    discriminator = build_discriminator_with_spectral_norm()

    discriminator.compile(optimizer=Adam(2e-4, beta_1=0.5), loss='mse', metrics=['accuracy'])

    # GAN model
    input_grayscale = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    generated_color = generator(input_grayscale)
    discriminator.trainable = False
    validity = discriminator(generated_color)

    gan = Model(input_grayscale, [generated_color, validity])
    gan.compile(optimizer=Adam(2e-4, beta_1=0.5), loss=[perceptual_loss, 'binary_crossentropy'], loss_weights=[100, 1])

    # Define the callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('colorization_model_best.keras', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    history = gan.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=len(X_val) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1
    )

# Plotting training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
   
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
   
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
   
    plt.show()

plot_history(history)

# Save the final model
gan.save('sar_colorization_final.keras')

# Load the best model
from tensorflow.keras.models import load_model
loaded_model = load_model('sar_colorization_final.keras')


import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import os

# Function to preprocess the test image
def preprocess_image(image_path, target_size=(256, 256)):
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    # Resize the image
    img = img.resize(target_size)
    # Convert to numpy array
    img = np.array(img)
    # Normalize the image (assuming normalization was done in the range [-1, 1])
    img = (img / 127.5) - 1
    # Expand dimensions to match generator input shape (batch_size, height, width, channels)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

# Function to denormalize the output image back to [0, 255] range
def denormalize_output(img):
    img = (img + 1) * 127.5  # Denormalize to [0, 255]
    img = np.clip(img, 0, 255)  # Clip to valid pixel range
    img = img.astype('uint8')  # Convert to unsigned integer type
    return img

# Load and preprocess the external test image
image_path = '{test_image}.jpg'  # Path to the external grayscale test image
test_image = preprocess_image(image_path)

# Use the generator to predict the colorized image
predicted_color_image = generator.predict(test_image)

# Denormalize the generated color image
predicted_color_image = denormalize_output(predicted_color_image[0])

# Visualize the input and output images
plt.figure(figsize=(10, 5))

# Display the grayscale input image
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(test_image), cmap='gray')  # Remove extra dimensions
plt.title("Grayscale Input Image")
plt.axis('off')

# Display the generated colorized image
plt.subplot(1, 2, 2)
plt.imshow(predicted_color_image)
plt.title("Generated Colorized Image")
plt.axis('off')

plt.show()
