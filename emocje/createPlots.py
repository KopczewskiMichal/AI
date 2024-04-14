import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import sys

def training_history(history):
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', color='grey')
    plt.legend()
    
    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', color='grey')
    plt.legend()

    # Save plot to file
    # filename = sys.argv[0].split('/')[-1]
    plt.savefig('plots/plot.png')
    plt.close()


def classification_examples():
  plt.figure(figsize=(10,10))
  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
      plt.xlabel(predicted_labels[i])
  plt.savefig('2.3.png')
  plt.show()




if __name__ == "__main__":
  print("Jest to plik pomocniczy, mający drukować wykresy.\n" +
  "Proszę uruchomić potrzebne funkcje z w innym pliku, podając odpowiednie parametry.")