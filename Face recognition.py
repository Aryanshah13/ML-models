#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt

saved_model_path = r"C:\Users\shaha\Downloads\archive (3)\model"
model = tf.keras.models.load_model(saved_model_path)

def img_to_embedding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

def get_user_image(name, database):
    if name in database:
        image_path = database[name]
        img = tf.keras.preprocessing.image.load_img(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print("User not found in the database.")
database = {
    "danielle": r"C:\Users\shaha\Downloads\archive (3)\images\danielle.png",
    "younes": r"C:\Users\shaha\Downloads\archive (3)\images\younes.jpg",
    "tian": r"C:\Users\shaha\Downloads\archive (3)\images\tian.jpg",
    "kian": r"C:\Users\shaha\Downloads\archive (3)\images\kian.jpg",
    "dan": r"C:\Users\shaha\Downloads\archive (3)\images\dan.jpg",
    "sebastiano": r"C:\Users\shaha\Downloads\archive (3)\images\sebastiano.jpg",
    "bertrand": r"C:\Users\shaha\Downloads\archive (3)\images\bertrand.jpg",
    "kevin": r"C:\Users\shaha\Downloads\archive (3)\images\kevin.jpg",
    "felix": r"C:\Users\shaha\Downloads\archive (3)\images\felix.jpg",
    "benoit": r"C:\Users\shaha\Downloads\archive (3)\images\benoit.jpg",
    "arnaud": r"C:\Users\shaha\Downloads\archive (3)\images\arnaud.jpg",
    "andrew": r"C:\Users\shaha\Downloads\archive (3)\images\andrew.jpg"
}

while True:
    name_input = input("Enter a user name (or type 'exit' to quit): ")
    if name_input.lower() == 'exit':
        break
    get_user_image(name_input, database)


# In[ ]:




