#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
import cv2
def create_siamese_network(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(input_img)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    encoded = Dense(128)(x)
    model = Model(input_img, encoded)
    return model
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
def load_and_preprocess_image(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  
    return np.expand_dims(image, axis=0)
rohan_image_path = "rohan.jpeg"
aryan_image_path = "ARYAN.jpeg"

rohan_image = load_and_preprocess_image(rohan_image_path)
aryan_image = load_and_preprocess_image(aryan_image_path)
siamese_network = create_siamese_network(input_shape)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
output_a = siamese_network(input_a)
output_b = siamese_network(input_b)
distance = Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)))([output_a, output_b])
siamese_model = Model(inputs=[input_a, input_b], outputs=distance)
siamese_model.compile(optimizer='adam', loss=contrastive_loss)
similarity_distance = siamese_model.predict([rohan_image, aryan_image])
similarity_threshold = 0.2 
if similarity_distance < similarity_threshold:
    print("The images are similar.")
else:
    print("The images are dissimilar.")


# In[ ]:




