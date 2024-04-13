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

images_width_height = 100

def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(images_width_height, images_width_height, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(8, activation='softmax'))
	# compile model
	# opt = SGD(learning_rate=0.001, momentum=0.9) # to możliwy optimizer
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model

def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

def run_test_harness():
	model = define_model()
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory('Dataset/train/',
		class_mode='categorical', batch_size=64, target_size=(images_width_height, images_width_height), color_mode='grayscale')
	test_it = datagen.flow_from_directory('Dataset/validation/',
		class_mode='categorical', batch_size=64, target_size=(images_width_height, images_width_height), color_mode='grayscale')

	checkpoint = ModelCheckpoint("best_model.keras", monitor='accuracy', verbose=1,
    save_best_only=True, mode='auto')

	# fit model
	print("Zaraz rozpoczynamy trenowanie")
	history = model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=1,
		callbacks=[checkpoint])
	# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
	print('> %.3f' % (acc * 100.0))

	# learning curves
	summarize_diagnostics(history)


if __name__ == '__main__':
  run_test_harness()