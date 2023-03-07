import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

# Set up Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = 'sajacky'
os.environ['KAGGLE_KEY'] = '5a36788e6cc079b7e25c0712f9f15beb'

# Download and unzip the dataset
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip chest-xray-pneumonia.zip -d dataset

# Define paths to the dataset
train_dir = 'dataset/chest_xray/train'
val_dir = 'dataset/chest_xray/val'
test_dir = 'dataset/chest_xray/test'

# Define the image size and batch size
img_size = 64
batch_size = 32

# Create a data generator for training data with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# Create a data generator for validation and test data
val_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

test_data = val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(8*8*256, input_dim=latent_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

latent_dim = 100
generator = build_generator(latent_dim)

def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

img_shape = (64, 64, 1)
discriminator = build_discriminator(img_shape)

# Training setup

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# Define the loss function for the discriminator
def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Define the loss function for the generator
def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

# Compile the discriminator and generator with their respective loss functions and optimizer
discriminator.compile(loss=discriminator_loss, optimizer=optimizer)
generator.compile(loss=generator_loss, optimizer=optimizer)

# Define the batch size and number of epochs
batch_size = 32
epochs = 100

# Create a function to generate noise vectors for the generator input
def generate_noise(batch_size, noise_dim):
    return tf.random.normal([batch_size, noise_dim])


# Define the number of epochs
epochs = 100

# Loop over the epochs
for epoch in range(epochs):
    
    # Loop over the batches in the dataset
    for image_batch in train_dir:
        
        # Generate noise
        noise = generate_noise(batch_size, latent_dim)
        
        # Generate fake images using the noise vector
        fake_images = generator.predict(noise)
        
        #MAke img to array and append it to a list
        image_batch = making_img_batch(image_batch)
        tf.ragged.constant(image_batch)
        # Concatenate the real and fake images
        combined_images = tf.concat([image_batch, fake_images], axis=0)
        
        # Create the labels for the discriminator
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        combined_labels = tf.concat([real_labels, fake_labels], axis=0)
        
        # Add noise to the labels
        combined_labels += 0.05 * tf.random.uniform(tf.shape(combined_labels))
        
        # Train the discriminator on the combined images
        discriminator_loss = discriminator.train_on_batch(combined_images, combined_labels)
        
        # Generate noise for the generator input
        noise = generate_noise(batch_size, latent_dim)
        
        # Train the generator
        generator_loss = gan.train_on_batch(noise, tf.ones((batch_size, 1)))
        
    # Print the losses
    print(f"Epoch {epoch+1}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")
