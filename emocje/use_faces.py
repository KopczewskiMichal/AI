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
  model = get_model()


def get_model():
  model = load_model("best_model.keras", compile=False)

  model.compile(
      optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
  )
  return model

def evaluate(model):
  datagen = ImageDataGenerator(rescale=1.0 / 255.0)
  test_it = datagen.flow_from_directory(
    "Dataset/validation/",
    class_mode="categorical",
    batch_size=64,
    target_size=(224, 224),
    color_mode="grayscale",
  )
  model.evaluate(test_it, verbose=1)

def predict_image(image_path: str) -> str:
  image = load_img(image_path, target_size=(224, 224))

  # Wersja własna, nie przewiduje sieci przystosowanej do cz-b
  # input_arr = img_to_array(image)
  # input_arr = np.array([input_arr])  # Convert single image to a batch.
  # predictions = model.predict(input_arr)
  # print(predictions)

  # Werjsa od chata
  gray_image = cv2.cvtColor(input_arr, cv2.COLOR_RGB2GRAY)
  input_arr = np.expand_dims(gray_image, axis=0)
  predictions = model.predict(input_arr)
  print(predictions)


if __name__ == "__main__":
  main()