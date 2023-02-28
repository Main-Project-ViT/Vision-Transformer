
import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,load_img,array_to_img
from pathlib import Path

import json
import zipfile
import os

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

pneumonia_folder ='/content/dataset/chest_xray/train/PNEUMONIA/'
path,dirs,files = next(os.walk(pneumonia_folder))
noOfPneumonia = len(files)

normal_folder ='/content/dataset/chest_xray/train/NORMAL/'
path,dirs,files = next(os.walk(normal_folder))
noOfNormal = len(files)
print(noOfPneumonia, noOfNormal)

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
    brightness_range=[0.5,1.5],
)
original_folder ='/content/dataset/chest_xray/train/NORMAL/'

#y=122
for y in range(0,1242):
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

import shutil

images = [f for f in os.listdir('/content/augmented2/')]


for image in images:
    new_path = '/content/dataset/chest_xray/train/NORMAL/'+image
    print(new_path)
    shutil.move('/content/augmented2/'+image, new_path)

