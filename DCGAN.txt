import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
latent_dim = 128
height = 64
width = 64
channels = 1
epochs = 30
batch_size = 64
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    directory='/path/to/train_data',
    target_size=(height, width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=None,
    subset='training')

validation_data = train_datagen.flow_from_directory(
    directory='/path/to/train_data',
    target_size=(height, width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=None,
    subset='validation')
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim)),
        layers.Dense(8 * 8 * 256),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
    ],
    name="generator",
)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(height, width, channels)),
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(256, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_output, name="gan")
gan.compile(loss='binary_crossentropy', optimizer='adam')
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    for i in range(len(train_data)):
        real_images = train_data[i]
        batch_size = real_images.shape[0]
        noise = np.random.randn(batch_size, latent_dim)
        fake_images = generator.predict(noise)

        # Train discriminator
        combined_images = np.concatenate([real_images, fake_images])
       
