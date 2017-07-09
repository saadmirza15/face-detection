from __future__ import absolute_import, division, print_function

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation



# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
## To Do. same as before.
    ## Define your network here
network = input_data(shape=[None, 100, 100, 3], 
                        data_preprocessing=img_prep, 
                        data_augmentation=img_aug) 

network = conv_2d(network, 64, 3, activation='relu') 
network = max_pool_2d(network, 2) 
network = conv_2d(network, 32, 3, activation='relu')  
network = max_pool_2d(network, 2) 
network = conv_2d(network, 32, 3, activation='relu')  
network = max_pool_2d(network, 2) 
network = fully_connected(network, 256, activation='relu') 
network = dropout(network, 0.5) 
network = fully_connected(network, 256, activation='relu') 
network = dropout(network, 0.5) 
network = fully_connected(network, 20, activation='softmax') 
network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) 
# Define model
## To Do
    ## Define model and assign network. Same as training.
model = tflearn.DNN(network, tensorboard_verbose=0)
# Load Model into model object
## To Do.
    ## Use the model.load() function
model.load('model/facedetection.tfl')

# load test images
from tflearn.data_utils import image_preloader
import numpy as np
# Load path/class_id image file:


data_dir = "test/"

# or use the mode 'floder'
X, Y = image_preloader(data_dir, image_shape=(100, 100), mode='folder',
                        categorical_labels=True, normalize=True,
                        files_extension=['.jpg', '.png'], filter_channel=True)


X = np.array(X)
Y = np.array(Y)
print('===========================',X.shape)
# predict test images label
y_pred = model.predict(X)


# Compute accuracy of trained model on test images
print ("Accuracy: ",np.sum(np.argmax(y_pred, axis=1) == np.argmax(Y, axis=1))*100/Y.shape[0],"%")
