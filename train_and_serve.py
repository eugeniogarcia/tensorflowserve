import sys

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

print('TensorFlow version: {}'.format(tf.__version__))

#Obtiene datos para el entrenamiento y test
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Procesa los datos antes de usarlos para entrenar
#Escala los valores entre 0.0 y 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

#Cambia las dimensiones
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

#Clases
class_names = ['Camiseta', 'Pantalon', 'Jersey', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota']

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

model = keras.Sequential([keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, strides=2, activation='relu', name='Conv1'),
						  keras.layers.Flatten(), 
						  keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')])

model.summary()

testing = False
epochs = 5

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nPrecision del test: {}'.format(test_acc))

#Vamos a guardar el modelo para usarlo con Tensor Flow serving

import tempfile

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
