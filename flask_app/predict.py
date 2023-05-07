import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers
import tensorflow_addons as tfa


class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.conv = tf.keras.layers.Conv2D(3, patch_size, strides=patch_size, name='conv_patches')

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    
# Linearly transform patches by projecting it into a
# vector of size `projection_dim` and also adds a learnable position
# embedding to the projected vector.
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


tf.keras.utils.get_custom_objects().update({'Patches': Patches})

# Define the custom_objects dictionary to map the custom layer to its name
custom_objects = {'PatchEncoder': PatchEncoder}
def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

tf.keras.utils.get_custom_objects()["f1_m"] = f1_m
tf.keras.utils.get_custom_objects()["recall_m"] = recall_m
tf.keras.utils.get_custom_objects()["precision_m"] = precision_m
with tf.keras.utils.custom_object_scope({'Patches': Patches}):
  model = tf.keras.models.load_model('./model/newmodel.h5', custom_objects=custom_objects)



def input_reshape(input):
    img = cv2.imread(input)
    img = cv2.resize(img, (288, 288))
    img = img / 255.0
    input_data = np.expand_dims(img, axis=0)
    predictions = model.predict(input_data)
    if predictions[0][0]>predictions[0][1]:
        return 'Normal'
    else:
        return 'Pneumonia'
    
