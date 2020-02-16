import sys

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


#Obtiene datos para el entrenamiento y test
fashion_mnist = keras.datasets.fashion_mnist

(_, _), (test_images, test_labels) = fashion_mnist.load_data()

#Procesa los datos antes de usarlos para entrenar
#Escala los valores entre 0.0 y 1.0
test_images = test_images / 255.0

#Cambia las dimensiones
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

#Clases
class_names = ['Camiseta', 'Pantalon', 'Jersey', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota']

def show(idx, title):
  plt.figure()
  plt.imshow(test_images[idx].reshape(28,28))
  plt.axis('off')
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})


import random

rando = random.randint(0,len(test_images)-1)
show(rando, 'Ejemplo de imagen: {}'.format(class_names[test_labels[rando]]))


import json
data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

import requests
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/moda/versions/1:predict', data=data, headers=headers)

respuesta=json.loads(json_response.text)['predictions']


for i in range(0,3):
  show(i,'Prediccion: {}. Valor correcto: {}'.format(class_names[test_labels[i]],class_names[np.argmax(respuesta[i])]))
