import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
!pip install kaggle

import os
import json
import zipfile
kaggle = {
    "username": "sajacky",
    "api_key": "5a36788e6cc079b7e25c0712f9f15beb",
    "on_kernel": False,
    "dataset": {
        "sample": "-",
        "full": "paultimothymooney/chest-xray-pneumonia"
        }
}

if kaggle["on_kernel"]:
  path_prefix = '/kaggle/working/'
else:
  path_prefix = '/content/'

# Download dataset
def download_dataset(which_dataset):
  data = {"username": kaggle["username"],"key": kaggle["api_key"]}
  with open('kaggle.json', 'w') as json_file:
      json.dump(data, json_file)

  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/
  !chmod 600 ~/.kaggle/kaggle.json
  kaggle_dataset = kaggle["dataset"][which_dataset]
  !kaggle datasets download -d $kaggle_dataset
  
  # Paths neeeds to be changed manually because of different directory structures, check with !ls
  if not os.path.isdir('dataset'):
    print("Unzipping... ")
    zip_ref = zipfile.ZipFile('chest-xray-pneumonia.zip', 'r')
    zip_ref.extractall('dataset')
    zip_ref.close()
    !rm -rf /content/dataset/chest_xray/chest_xray
    !rm -rf /content/dataset/chest_xray/__MACOSX
    !rm -rf /content/chest-xray-pneumonia.zip
  

  #!ls Data


download_dataset("full")

data_files = os.listdir("dataset/chest_xray")
print(data_files)
def resample_data(move_from, move_to, cl, images_to_move=100):
  path = path_prefix + 'dataset/chest_xray/'

  classes = os.listdir(path + move_from)

  cl += '/'
  curr_path = path + move_from + cl
  for _, _, files in os.walk(curr_path):
    random.shuffle(files)
    files_to_move = files[:images_to_move]
    for fn in files_to_move:
      shutil.move(curr_path + fn, path + move_to + cl + fn)
      #print('Moved ' + curr_path + fn)
        
  print('Resampled Images')

move_from, move_to = 'train/', 'val/'
#resample_data(move_from, move_to, 'PNEUMONIA', 2534)



# Training images
print('Number of NORMAL training images:')
!ls /content/dataset/chest_xray/train/NORMAL/ | wc -l
print('Number of PNEUMONIA training images:')
!ls /content/dataset/chest_xray/train/PNEUMONIA/ | wc -l
print()


# Test images 
#resample_data('test/', 'val/', 'PNEUMONIA', 2690)
print('Number of NORMAL test images:')
!ls /content/dataset/chest_xray/test/NORMAL/ | wc -l
print('Number of PNEUMONIA test images:')
!ls /content/dataset/chest_xray/test/PNEUMONIA/ | wc -l

def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Compile the discriminator and generator models
from tensorflow.keras.optimizers import Adam
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Define the number of epochs and batch size
epochs = 10000
batch_size = 128
latent_dim = 100

# set path to the directory containing training data
train_dir = '/content/dataset/chest_xray/train/PNEUMONIA'
image_shape=[28,28,1]

# set path to the training data directory
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=image_shape[:2], batch_size=batch_size, class_mode='binary')
X_train, y_train = train_generator.next()

# Define the number of batches per epoch
batch_count = int(X_train.shape[0] / batch_size)

# Define the noise vector to generate fake images
noise = np.random.normal(0, 1, (batch_size, latent_dim))

# Define lists to keep track of the loss and accuracy of the discriminator and generator
d_loss_history = []
d_acc_history = []
g_loss_history = []

# Define a function to generate a grid of images from the generator
def generate_images(generator, epoch):
  # Generate fake images from the generator
  fake_images = generator.predict(noise)

  # Rescale the pixel values from [-1, 1] to [0, 1]
  fake_images = 0.5 * fake_images + 0.5

  # Create a grid of images
  fig, ax = plt.subplots(8, 8, figsize=(8, 8))
  #print(fake_images.shape)
  for i, ax_i in enumerate(ax.flatten()):
    # Get the i-th fake image
    img = fake_images[i,:, 0]
    
    # Plot the image on the i-th subplot
    ax_i.imshow(img, cmap='gray')
    ax_i.axis('off')
  plt.show()
  
  # Save the grid of images to a file
  fig.savefig('generated_images_epoch_%d.png' % epoch)

# Iterate over the epochs
for epoch in range(epochs):
  # Initialize the loss and accuracy of the discriminator and generator
  d_loss = 0
  d_acc = 0
  g_loss = 0
  
  # Iterate over the batches
  for batch_index in range(batch_count):
    # Select a batch of real images
    real_images = X_train[batch_index * batch_size:(batch_index + 1) * batch_size]

    # Generate a batch of fake images from the generator
    fake_images = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))

    # Train the discriminator on the batch of real images
    d_loss_real, d_acc_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))

    # Train the discriminator on the batch of fake images
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))

    # Compute the average loss and accuracy of the discriminator
    d_loss = (d_loss_real + d_loss_fake) / 2
    d_acc = (d_acc_real + d_acc_fake) / 2

    # Train the generator to fool the discriminator
    g_loss = gan.train_on_batch(np.random.normal(0, 1, (batch_size, latent_dim)), np.ones((batch_size, 1)))

  # Print the loss and accuracy of the discriminator and generator for each epoch
  print('Epoch %d: d_loss = %f, d_acc = %f, g_loss = %f' % (epoch, d_loss, d_acc, g_loss))

  # Add the loss and accuracy of the discriminator and generator to the history lists
  d_loss_history.append(d_loss)
  d_acc_history.append(d_acc)
  g_loss_history.append(g_loss)

  # Generate a grid of images from the generator and save it to a file every 100 epochs
  if epoch % 100 == 0:
    generate_images(generator, epoch)
