import sys
sys.path.append('..')
from keras.models import load_model
from train import define_data_generator
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def main():
  model = load_model('../emotions.keras', compile=False)
  model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

  data_generator = define_data_generator("../Dataset/test")

  compute_confusion_matrix(model, data_generator)


def compute_confusion_matrix(model, generator):
  predicted_labels = np.argmax(model.predict(generator), axis=1)

  true_labels = generator.classes

  conf_matrix = confusion_matrix(true_labels, predicted_labels)

  plt.figure(figsize=(8, 6))
  plt.imshow(conf_matrix, cmap=plt.cm.Blues)
  plt.colorbar()
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.xticks(np.arange(len(generator.class_indices)), generator.class_indices.keys(), rotation=45)
  plt.yticks(np.arange(len(generator.class_indices)), generator.class_indices.keys())
  plt.tight_layout()
  plt.savefig("confusion_matrix.png")
  plt.show()


if __name__ == '__main__':
  main()