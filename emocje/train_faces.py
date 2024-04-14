import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

import createPlots

images_width_height = 224
batch_size = 64


def define_model():
    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                padding="same",
                input_shape=(images_width_height, images_width_height, 1),
            ),
            MaxPooling2D((2, 2)),
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                padding="same",
            ),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu", kernel_initializer="he_uniform"),
            Dense(8, activation="softmax"),
        ]
    )
    # compile model
    # opt = SGD(learning_rate=0.001, momentum=0.9) # to możliwy optimizer
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model


def run_test_harness():
    model = define_model()
    datagen = ImageDataGenerator(rescale=1.0 / 255.,
        horizontal_flip=True,
    )
    # prepare iterators
    train_it = datagen.flow_from_directory(
        "Dataset/train/",
        class_mode="categorical",
        batch_size=batch_size,
        target_size=(images_width_height, images_width_height),
        color_mode="grayscale",
        shuffle=True,
    )
    test_it = datagen.flow_from_directory(
        "Dataset/validation/",
        class_mode="categorical",
        batch_size=batch_size,
        target_size=(images_width_height, images_width_height),
        color_mode="grayscale",
        shuffle=True,
    )

    checkpoint = ModelCheckpoint(
        "best_model.keras",
        monitor="accuracy",
        verbose=1,
        save_best_only=True,
        mode="auto",
    )

    # fit model
    print(len(train_it))
    history = model.fit(
        train_it,
        # steps_per_epoch=len(train_it),
        steps_per_epoch=len(train_it),
        validation_data=test_it,
        validation_steps=len(test_it),
        epochs=40,
        verbose=1,
        callbacks=[checkpoint],
    )
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
    print("> %.3f" % (acc * 100.0))

    createPlots.training_history(history)


if __name__ == "__main__":
    run_test_harness()
