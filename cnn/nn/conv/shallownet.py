from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
	@staticmethod
	def build(width, height,depth, classes): #total # of classes our network should predict

		#init model with input shape to be channels last (H,W,C)
		model = Sequential()
		inputShape = (height,width,depth)

		if K.image_data_format() == "channels_first":
			inputShape = (depth,height,width) #load channel first or last


		model.add(Conv2D(32,(3,3), padding = "same", input_shape = inputShape)) #define the first/only Convolution to RELU layer
		#32 filters (K) that are 3x3 (FxF). Same padding ensures the size of output of convolution is the same as the input.
		model.add(Activation("relu")) #Adds reLU activation	
		
		model.add(Flatten()) #flatten to 1D list
		model.add(Dense(classes)) #Dense layer uses same # of nodes as output class
		model.add(Activation("softmax")) #activation function to finally give class label probabilities.

		return model

		
