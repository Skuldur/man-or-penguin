import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
import os, glob
K.set_image_dim_ordering('th')

if __name__ == "__main__":
	model = load_model('penguin_model_100_epochs.h5')

	test_data_dir = "images/eval"
	test_datagen = ImageDataGenerator(rescale=1./255)

	validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(150,150),
        batch_size=32,
        class_mode='binary')

	scores =  model.evaluate_generator(validation_generator, 400)
	print("Accuracy: %.2f%%" % (scores[1]*100))
