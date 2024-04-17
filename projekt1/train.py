import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.shedules import ExponentialDecay

target_size = (48, 48)

def main():
  train_datagen = define_data_generator('Dataset/train')
  val_datagen = define_data_generator('Dataset/test')

  model = define_model()

  checkpoint = ModelCheckpoint("emotions.keras", monitor='accuracy', verbose=1,
    save_best_only=True, mode='auto')

  history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=40,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
    verbose=1,
    callbacks=[checkpoint]
)

  
def define_data_generator(source_dir: str):
  datagen = ImageDataGenerator(rescale=1./255)
  generator = datagen.flow_from_directory(
          train_dir,
          target_size=target_size,
          batch_size=64,
          color_mode="grayscale",
          class_mode='categorical')
  return generator


def define_model() ->  Model:
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
  model.add(Dense(7, activation='softmax'))

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model



if __name__ == '__main__':
  main()