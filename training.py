from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import cv2
from glob import glob
import matplotlib.pyplot as plt


data_directory = "train"
# Useful for getting our categories
folders = glob('train/*')
categories = [x.replace('train','').strip('\\\\') for x in folders]


def train_models(categories=categories, data_directory=data_directory):
	for category in categories:
			path = os.path.join(data_directory, category)
			for img in os.listdir(path):
					img_array = cv2.imread(os.path.join(path,img))
					break
			break

	# Re-size all the images to this
	IMAGE_SIZE = [224, 224]

	train_path = 'train'
	valid_path = 'test'

	# Import the VGG16 library as shown below and add preprocessing layer to the front of VGG
	# Here we will be using imagenet weights

	vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

	# Don't train existing weights
	for layer in vgg16.layers:
			layer.trainable = False

	# Our layers - you can add more if you want
	x = Flatten()(vgg16.output)
	prediction = Dense(len(categories), activation='softmax')(x)

	# Create a model object
	model = Model(inputs=vgg16.input, outputs=prediction)

	# View the structure of the model
	model.summary()

	# Tell the model what cost and optimization method to use. 
	# There are a wide array of metrics available, see here for more: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Use the Image Data Generator to import the images from the dataset
	train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
	test_datagen = ImageDataGenerator(rescale = 1./255)

	# Make sure you provide the same target size as initialied for the image size
	training_set = train_datagen.flow_from_directory('train', target_size = (224, 224), batch_size = 32, class_mode = 'categorical')
	test_set = test_datagen.flow_from_directory('test', target_size = (224, 224), batch_size = 32, class_mode = 'categorical')
	checkpoint_filepath = 'models/{epoch:02d}-{val_loss:.2f}.hdf5'
	model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=False)

	# fit the model
	# Run the cell. It will take some time to execute
	r = model.fit(training_set, callbacks=[model_checkpoint_callback], validation_data=test_set, epochs=6, steps_per_epoch=len(training_set), validation_steps=len(test_set))
	y_pred = model.predict(test_set)




train_models(categories=categories, data_directory=data_directory)