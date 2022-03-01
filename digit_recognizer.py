#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import argmax
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image, ImageOps

print(tf.__version__)


# In[2]:


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train = tf.keras.utils.normalize(x_train, axis=1)
print(x_train.shape)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)


# In[3]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


# In[4]:


# save model
model.save('digit_model.model')

# load the model saved
new_model = tf.keras.models.load_model('digit_model.model')


# In[5]:


predictions = new_model.predict([x_test])


# In[6]:


print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()


# In[7]:


# load and prepare the image
def load_image(filename):
    # load the image
    img = ImageOps.grayscale(Image.open(filename))

    # reshape into a single sample with 1 channel
    img = np.resize(img, (28,28,1))

    # convert to array
    img = img_to_array(img)
    img = img.reshape(1, 28, 28)

    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# In[8]:


img = load_image('tests/img2.png')

predict = model.predict(img)
for ind, res in enumerate(predict[0]):
    print(f'Número {ind} - Chance {res * 100:.5f}%')
print('O número da imagem é:',argmax(predict))

plt.imshow(img[0])
plt.show()

