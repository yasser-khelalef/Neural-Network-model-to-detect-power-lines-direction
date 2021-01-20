import warnings, logging, os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import pickle, PIL
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import plot_model
from keras import backend as K
from tensorflow.keras import backend
from keras.callbacks import History
from math import gcd
from os import walk

#Change these parameters to fit your needs

#Saving and Loading Settings
save_model = 1 #Save the model? True or False
load_old = 0 #Load a model to continue training? True or False. False will result in a new model being created.
netpathdir = '/home/yasser/Downloads/Pics' #The directory to load/save the neural net
h5name = 'NeuralNetwork' #the name of the neural net

#Neural Network Image Input Settings 
img_width = 80 #Size of the neural network input width
img_height = 60 #Size of the neural network input height
grayscale = 1 #Do you want the images to be grayscale when they go in the neural net? True or False 
imgpathdir = '/home/yasser/Downloads/Pics' #Directory to the folders containing the training image sets

#Neural Network Shape Parameters
dense_width = 32 #How wide the Dense layer will be
conv_width = 3 #How many channels the Conv2D layers will have
p1 = (2,1) #Max Pooling Kernel 1
p2 = (2,3) #Max Pooling Kernel 2
#c1 = (int(img_width/10), int(img_height/10)) #Conv2D Kernel 1
c1 = (4,4)
c2 = (4,4) #Conv2D Kernel 2
g=gcd(int(img_width), int(img_height))
#strides1=(int(img_width/g), int(img_height/g)) #Stride for Conv2D 1
strides1=(1,1)
strides2=(1,1) #Stride for Conv2D 2

#Neural Network Training Parameters
nb_train_samples = 20 #Number of training images per epoch
nb_validation_samples = 10 #Number of validation images per epoch
epochs = 250 #Number of epochs to train for
batch_size = 10 #Size of each batch
dropout=0.5 #Dropout Percent (0.5 is almost always optimal)





#Code begins
h5name+='.h5'
f=os.path.join(netpathdir,h5name)
train_data_dir = imgpathdir
validation_data_dir = imgpathdir
for (_, n, _) in walk(train_data_dir):
	break
num_outs=len(n)
color='rgb'
if grayscale:
	input_shape = (img_width, img_height, 1)
	color='grayscale'
elif K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
if not load_old:
	model = Sequential()
	model.add(Conv2D(conv_width, c1, strides=strides1, input_shape=input_shape))
	#model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=p1))
	model.add(Conv2D(conv_width, c2, strides=strides2))
	#model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=p2))
	model.add(Flatten())
	model.add(Dense(dense_width))
	#model.add(Activation('relu'))
	model.add(Dropout(dropout))
	model.add(Dense(num_outs))
	model.add(Activation('relu'))
	model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
	#model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
	#model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
else:
	try:
		model = load_model(f)
	except:
		print('The name or directory is probably incorrect or invalid.')
model.summary()

train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='categorical',
	color_mode=color,
	shuffle=True)

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='categorical', 
	color_mode=color,
	shuffle=True)

model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size
	)

model.summary()
if save_model:
	model.save(f)	
		

		

