#augmentation

import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,load_img,array_to_img
from pathlib import Path

path,dirs,files = next(os.walk('/content/dataset/chest_xray/train/NORMAL/'))
file_count = len(files)
print(file_count)

if not os.path.exists('/content/augmented2/'):
    os.makedirs('/content/augmented2/')

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
original_folder ='/content/dataset/chest_xray/train/NORMAL/'

#y=122
for y in range(0,1341):
  filename = os.listdir(original_folder)[y]
  img_path = original_folder+filename

  img = load_img(img_path)
  x = img_to_array(img)
  x = x.reshape((1,) + x.shape)

  for batch in datagen.flow(x, batch_size=1, save_to_dir='/content/augmented2/', save_prefix='NORMAL_A', save_format='jpeg'):
    break
  
  y+=1
  print(y)

path,dirs,files = next(os.walk('/content/augmented2/'))
file_count = len(files)
print(file_count)




#copying images to folders

import shutil

images = [f for f in os.listdir('/content/augmented2/')]


for image in images:
    new_path = '/content/dataset/chest_xray/train/NORMAL/'+image
    print(new_path)
    shutil.move('/content/augmented2/'+image, new_path)
