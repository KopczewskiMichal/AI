import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.optimizers import Adam
from keras.optimizers import schedules
from keras.layers import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf #! wypadałoby zoptymalizować import

target_size = (48, 48) # 48x48 to rozmiar obrazków w orginalnym datasecie
epochs = 60

def main():
  train_generator = define_data_generator('betterDataset/train')
  validation_generator = define_data_generator('betterDataset/test')

  model = define_model()

  checkpoint = ModelCheckpoint("emotions-better2.keras", monitor='accuracy', verbose=1,
    save_best_only=True, mode='auto', save_freq='epoch')
  early_stop = EarlyStopping(monitor='accuracy', patience=6)

  history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
    verbose=1,
    callbacks=[checkpoint, early_stop]
)

  
def define_data_generator(source_dir: str):
  datagen = ImageDataGenerator(rescale=1./255)
  generator = datagen.flow_from_directory(
          source_dir,
          target_size=target_size,
          batch_size=32,
          color_mode="grayscale",
          class_mode='categorical')
  return generator


def define_model():
  model = Sequential()

  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=target_size + (1,))) # 1 oznacza że mamy 1 kanał (bw)
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(8, activation='softmax'))

  learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,  
    decay_rate=0.9  
)

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model



if __name__ == '__main__':
  main()