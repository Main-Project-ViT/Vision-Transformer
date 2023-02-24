import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,load_img,array_to_img
from pathlib import Path
import os

path,dirs,files = next(os.walk('/content/dataset/chest_xray/train/NORMAL/'))
file_count = len(files)
print(file_count)

os.mkdir('/content/augmented1/')

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest'
)
original_folder ='/content/dataset/chest_xray/train/NORMAL/'

#y=122
for y in range(300,599):
  filename = os.listdir(original_folder)[y]
  img_path = original_folder+filename

  img = load_img(img_path)
  x = img_to_array(img)
  x = x.reshape((1,) + x.shape)

  i = 0
  for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='/content/augmented1/', save_prefix='NORMAL', save_format='jpeg'):
    i += 1
    if i > 2:
       break

  y+=1
  print(y)

path,dirs,files = next(os.walk('/content/augmented1/'))
file_count = len(files)
print(file_count)