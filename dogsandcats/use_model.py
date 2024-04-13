import sys

from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import img_to_array
from keras.utils import load_img
from tensorflow.keras.preprocessing import image
import numpy as np
import os



def main():
  from keras.models import load_model

  model = load_model('best_model.keras', compile=False)

  model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

  # model.summary()
  # model = re_train(model)
  # model.summary()
  predict_image(model, "./dogs-vs-cats/test1/2492.jpg")
  predict_image(model, "./dogs-vs-cats/test1/2494.jpg")
  predict_image(model, "./dogs-vs-cats/test1/2495.jpg")
  evaluate(model)


def predict_image(model, image_path:str): 
  image = load_img(image_path, target_size=(200, 200))
  input_arr = img_to_array(image)
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  # input _arr /= 255.0
  predictions = model.predict(input_arr)
  print(predictions)   # 0 kot, 1 pies

# def predict_images_dir(model, images_dir):
#   def create_batch(images_path_arr):
#     result = np.zeros((len(images_path_arr), 200, 200, 3))
#     for index, image_path in enumerate(images_path_arr):
#       image = load_img(images_dir + image_path, target_size=(200, 200))
#       img_arr = img_to_array(image)
#       result[index] = img_arr
#     return result
  
  # def predict_bath_results(batch):
  #   def save_bath_result_to_file(filepath:str, data):
  #     print(data)
  #     file = open(filepath, "a")
  #     file.write(str(data))
  #     file.close()
  #   predictions = model.predict(batch)
  #   save_bath_result_to_file("predictions.txt", predictions)

  # # batch_size = 64
  # # images_paths = os.listdir(images_dir)

  # for i in range(0, len(images_paths), batch_size):
  #   if i <= len(images_paths) - batch_size:
  #     predict_bath_results(create_batch(images_paths[i:i+batch_size]))
  #   else:
  #     predict_bath_results(create_batch(images_paths[i:]))

    # print(predictions)


def evaluate(model):
  datagen = ImageDataGenerator(rescale=1.0/255.0)
  test_it = datagen.flow_from_directory('dogs-vs-cats/validation/',
    class_mode='binary', batch_size=64, target_size=(200, 200))
  model.evaluate(test_it, verbose=1)

# def re_train(model): # co ciekawe użycie tej metody jest bezsensowne i mocno pogarsza wynik, tak jakby trenowało od zera model
#   datagen = ImageDataGenerator(rescale=1.0/255.0)
# 	# prepare iterators
#   train_it = datagen.flow_from_directory('dogs-vs-cats/train/',
# 		class_mode='binary', batch_size=64, target_size=(200, 200))
#   test_it = datagen.flow_from_directory('dogs-vs-cats/validation/',
# 		class_mode='binary', batch_size=64, target_size=(200, 200))
# 	# fit model
#   model.fit(train_it, steps_per_epoch=len(train_it),
# 		validation_data=test_it, validation_steps=len(test_it), epochs=2, verbose=1)
  

  return model


if __name__ == '__main__':
  main()