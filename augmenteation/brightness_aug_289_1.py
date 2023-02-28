#augmentation to normal

import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,load_img,array_to_img
from pathlib import Path

path,dirs,files = next(os.walk('/content/dataset/chest_xray/train/NORMAL/'))
file_count = len(files)
print(file_count)

if not os.path.exists('/content/augmentedNormal/'):
    os.makedirs('/content/augmentedNormal/')

datagen = ImageDataGenerator(
    brightness_range=[0.5,1.5],
)
original_folder ='/content/dataset/chest_xray/train/NORMAL/'

#y=122
for y in range(0,1000):
  filename = os.listdir(original_folder)[y]
  img_path = original_folder+filename

  img = load_img(img_path)
  x = img_to_array(img)
  x = x.reshape((1,) + x.shape)

  for batch in datagen.flow(x, batch_size=1, save_to_dir='/content/augmentedNormal/', save_prefix='NORMAL_A', save_format='jpeg'):
    break
  
  y+=1
  print(y)

path,dirs,files = next(os.walk('/content/augmentedNormal/'))
file_count = len(files)
print(file_count)


##moving augmented images to folders

import shutil

images = [f for f in os.listdir('/content/augmentedNormal/')]

for image in images:
    new_path = '/content/dataset/chest_xray/train/NORMAL/'+image
    print(new_path)
    shutil.move('/content/augmentedNormal/'+image, new_path)



#augmentation to pneumonic

import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,load_img,array_to_img
from pathlib import Path

path,dirs,files = next(os.walk('/content/dataset/chest_xray/train/PNEUMONIA/'))
file_count = len(files)
print(file_count)

if not os.path.exists('/content/augmentedPneumonia/'):
    os.makedirs('/content/augmentedPneumonia/')

datagen = ImageDataGenerator(
    brightness_range=[0.5,1.5],
)
original_folder ='/content/dataset/chest_xray/train/PNEUMONIA/'

#y=122
for y in range(0,1000):
  filename = os.listdir(original_folder)[y]
  img_path = original_folder+filename

  img = load_img(img_path)
  x = img_to_array(img)
  x = x.reshape((1,) + x.shape)

  for batch in datagen.flow(x, batch_size=1, save_to_dir='/content/augmentedPneumonia/', save_prefix='NORMAL_A', save_format='jpeg'):
    break
  
  y+=1
  print(y)

path,dirs,files = next(os.walk('/content/augmentedPneumonia/'))
file_count = len(files)
print(file_count)


#moving augmented images to PNEUMONIA

import shutil

images = [f for f in os.listdir('/content/augmentedPneumonia/')]


for image in images:
    new_path = '/content/dataset/chest_xray/train/PNEUMONIA/'+image
    print(new_path)
    shutil.move('/content/augmentedPneumonia/'+image, new_path)