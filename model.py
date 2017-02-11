
# coding: utf-8
# Israel Barrón - MX - Feb 2016
# Behavioral Clonign Project for Self Driving Car ND

# # Behavioral Cloning
# 


# Import some basic libraries
import math
import numpy as np
import random


# ## Choosing the Data
# 
# In order to get the data, we need to capture a "smooth" running in the simulator. It is important to get smooth dataset because if the training data has samples where the car is zigzaggint, that behavior will also be cloaned by the network (and will be even worse). A good data set will keep the car in the center of the road most of the time. 
# Even though we do not want pictures that teach our car to go to the lanes, it is important to record some pictures where the car in on the lanes and steering in order to go back to the center, this will teach the car to recover from situation where it goes over the lanes. This is achieved by intentionally steering the car over the lanes and recording only when the car is going back to the center.  
# 
# The original number of samples is 8036 from a smooth drive + 920 for recovery.   
# 
# *Note: the number of samples mentiones above, is before data augmentation*


from skimage import io
import pandas as pd

#Start by importing the data from folder and the labels from csv file.
## Load new pics image collection
X_train = io.imread_collection('./session_recover/IMG/center*.jpg:./udacity_session/data/IMG/center*.jpg')
X_train = np.array(io.concatenate_images(X_train))

# Load labels
df=pd.read_csv('./session_recover/driving_log.csv', sep=',',header=0, usecols=['steering'])
y_train_or = np.array(df.values)
df=pd.read_csv('./udacity_session/data/driving_log.csv', sep=',',header=0, usecols=['steering'])
y_train_or = np.concatenate((y_train_or, np.array(df.values)), axis=0)


# ## Preprocess the Data
# 
# For data preprocessing, the pictures are cropped in order to remove the background and keep the road. Three data augmentation techniques are used, resulting in 3 training datasets.
# The techniques applied are:  
# - Adaptative histogram equialization.  
# - Adaptative histogram equialization + random_rotations.  
# - Featurewise.
# 
# For validation dataset, Adaptative histogram equialization + random_rotations is applied.  
# Normalization is not applied here, the normalization is applied per batch during training.
# 
# Here is the justification for the choosen techniques:  
# 
# **Adaptive histogram equalization**  
# Adaptive histogram equalization (AHE) is a computer image processing technique used to improve contrast in images. It differs from ordinary histogram equalization in the respect that the adaptive method computes several histograms, each corresponding to a distinct section of the image, and uses them to redistribute the lightness values of the image. It is therefore suitable for improving the local contrast and enhancing the definitions of edges in each region of an image.
# https://en.wikipedia.org/wiki/Adaptive_histogram_equalization  
#   
# **Feature-wise**  
# Feature wise data augmentation is applied in the following ways:  
# *featurewise_center*: Set input mean to 0 over the dataset, feature-wise.   
# *featurewise_std_normalization*: Divide inputs by std of the dataset, feature-wise.  
# Refer to keras documentation for more details: 
# https://keras.io/preprocessing/image/  
# 
# 
# **Adaptive histogram equalization + random rotations**  
# Combining AHE with random rotations help the model to adapt to different perspectives that are not in the previous sets, this helps the model to adapt to never seen before situations (it helps to generalize).
# 
# 
# Note: These preprocessing techniques are not applied when testing in the simulator (except cropping).  
# 



# Function crops the pictures, removing the landscape and keeping the road
def cropPicture(X_train):
    aux=[]

    for img in X_train:
        img=img[60:160, 0:320]
        aux.append(img)
       
    return np.array(aux)






#Functions shuffle and split data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def shuffleInput(X_train, y_train):
    #Shuffle training data
    X_train, y_train = shuffle(X_train, y_train)
    #Splitting into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.2)
    return X_train, X_val, y_train, y_val




# Data augmentation function 
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy import ndimage
from skimage import exposure

def dataAugmentation(X_train, X_val, y_train, y_val, sufix):


    # Saving the original data set just in case we want to further apply other data augmentation filters
    np.save("y_train_orig"+sufix, y_train)
    np.save("X_train_orig"+sufix, X_train)


    #Applying equalizer to training set
    aux=[]
    for image in X_train:
        image = exposure.equalize_adapthist(image, clip_limit=0.03)
        aux.append(image)

    X_train =np.array(aux)
    aux=[]
    # Saving the dataset after data augmentation. So we can load the data in the future without
    # wasting time reprocessing it
    np.save("X_train_equ"+sufix, X_train)


    #Creating a second dataset with random rotations
    #aux=[]
    for image in X_train:
        image = ndimage.rotate(image, random.randint(-30, 30), reshape=False)
        aux.append(image)
    # Saving augmented set in disk
    np.save("X_train_dos"+sufix, np.array(aux))

    aux=[]

    #Creating a third dataset with featurewise
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    datagen.fit(X_train)
    # datagen.flow is an infinite loop, so we need to break the loop
    for X_trainx, y_trainx in datagen.flow(X_train, y_train, batch_size=len(X_train)):
        # Saving augmented set in disk  
        np.save("X_train_wise"+sufix, X_trainx)
        break
    #Since data augmentation uses too much memory, let's clean these variables
    datagen = 0
    X_trainx, y_trainx = 0,0  


    #Applying equalizer to validation set
    #aux=[]
    for image in X_val:
        image = exposure.equalize_adapthist(image, clip_limit=0.03)
        randomNumber=random.randint(0,5)
        if randomNumber>= 3:
            image = ndimage.rotate(image, random.randint(-30, 30), reshape=False)
        aux.append(image)
    # Saving validation set in disk  
    np.save("X_val_or"+sufix, np.array(aux))
    np.save("y_val_or"+sufix, y_val)
    
    
    aux=[]


#Calling function to preprocess data
X_train, X_val, y_train, y_val = shuffleInput(cropPicture(X_train), y_train_or)
dataAugmentation(X_train, X_val, y_train, y_val, sufix="")
print('data center augmentation finished')



# ## Designing the Network  
# The network is inspired by Nvidia's architecture, but it was adapted empirically for this project. The final model conserves some of the characteristics of Nvidia's architecture but some changes were applied for the specific needs of this problem; such as number of layers, kerner size and strides. Again, the final model is mainly the result of empirical practice.  
# Please refer to Nvidia's paper for more details about their aproach http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf  
# 
# As opposed to previous deep learning projects, this problem does not fit into a categorial one. Since we are trying to predict a single value given an input picture, this problem falls into the category of regression problems. For such reason, we need an architecture that fits a regression solution. 
# For categorical problems, the otput of the network is the same as the number of categories. In the case of regression, the output is a single neuron which yields the predicted value (steering angle in this case). For that reason, we discard one hote encoding this time.
# 
# ## Network Architecture  
# The architecture has 7 Convolutinal Layers and 3 fully connected layers.  
# Layers are distributed as follows:  
#   
# **L1 Convolutional** with 3x3 kernel and 3x3 Stride. Input =3@100x320  Output = 24@32x106.  
# **L2 Convolutional** with 3x3 kernel and 2x2 Stride. Output = 36@15x52.  
# **L3 Convolutional** with 1x1 kernel.                Output = 36@15x52.  
# **L4 Convolutional** with 3x3 kernel and 2x2 Stride. Output = 48@6x25.  
# **L5 Convolutional** with 1x1 kernel.                Output = 48@6x25.  
# **L6 Convolutional** with 3x3 kernel and 2x2 Stride. Output = 64@2x11.  
# **L7 Convolutional** with 1x1 kernel.                Output = 64@2x11.  
# **Flatten**                                          1408.  
# **L8 Fully connected**                               200.  
# **L9 Fully connected**                               100.  
# **L10 Fully connected**                              10.  
# **Output.**  
# 
# ## Training the Network  
# 
# **Samples**   
# The total number of samples is 7164 for training and 1792 for evaluation.  
#   
# **Normalization**  
# Normalization is applied per batch at the begining and again in L6. There was no normalization per dataset before training.   
# 
# **Activation**  
# As activation function, ELU is used in all the layers due to it's slightly better convergence than RELU. For a comparison between ELU and RELU please refer to this post for more details:  
# http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/  
# 
# **Regularization**  
# As a method to prevent overfitting, **dropout** layers are used with a value of 0.5 in layers L4, L6-L10. **Dropout** has proven to be very efficient for preventting overfitting and at the same time it's really simple.  
#   
# **Optimizer and loss operation**  
# As optimizer "adam" is used, as it has proven to be very efficient for regression problems.  
# We are dealing with a regression problem, so we need a loss operation that is proper for this kind of problems. A good choice is "mse", since it measures the average of the squares of the errors or deviations—that is, the difference between the estimator and what is estimated. (https://en.wikipedia.org/wiki/Mean_squared_error).  
# Weights are initialized in a normalized way in fully connected layers; this also improves the convergence.
# 
# **Fine-tuning**  
# Instead of mixing the different data augmentations datasets into a single one, I choose to train the model with a one by one approach, which is working as fine-tuning over the previous dataset and taking advantage of characteristics identified in the previous session and just adapting the weigths in the final training. That is why it is important to keep at the end the dataset that is more similar to the real scenario pictures, which in this case is the one with AHE.
# The number of epochs per dataset is as follows:
# 
# *Feature-wise*: --------------> 2 epochs.  
# *AHE + randon rotations*: -> 7 epochs.    
# *AHE*: -------------------------> 7 epochs.  
# 

# Architecture based on Nvidia's model but adapted to this particualar problem
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Merge
from keras.layers import Conv2D, Flatten, Dropout, MaxPooling2D,  BatchNormalization
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasRegressor

#Dropout value
dropVal = 0.5

def baseline_model():
    
    # L1
    model = Sequential()
    model.add(Conv2D(24, 3, 3, input_shape=(X_train[0].shape)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D((3,3)))
    
    #L2
    model.add(Conv2D(36, 3, 3))
    model.add(Activation('elu'))
    model.add(MaxPooling2D((2,2)))
    # L3
    model.add(Conv2D(36, 1, 1))
    model.add(Activation('elu'))
    # L4
    model.add(Conv2D(48, 3, 3))
    model.add(Activation('elu'))
    model.add(MaxPooling2D((2,2)))
    model.add((Dropout(dropVal)))
    # L5
    model.add(Conv2D(48, 1, 1))
    model.add(Activation('elu'))
    # L6
    model.add(Conv2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D((2,2)))
    model.add((Dropout(dropVal)))
    #L7
    model.add(Conv2D(64, 1, 1))
    model.add(Activation('elu'))
    model.add((Dropout(dropVal)))
    #L8
    model.add(Flatten())
    model.add(Dense(200, activation='elu', init='normal'))
    model.add((Dropout(dropVal)))
    #L9
    model.add(Dense(100, activation='elu', init='normal'))
    model.add((Dropout(dropVal)))
    #L10
    model.add(Dense(10, activation='elu', init='normal'))
    model.add((Dropout(dropVal)))
    model.add(Dense(1))
    model.summary()
    
    model.compile(optimizer='adam', loss='mse')
    return model




# ## Running the model

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=2, shuffle=True, batch_size=100, verbose=1)
# Load the "labels" for training data
y_train = np.load("y_train_orig.npy")
# Load validation set 
X_val = np.load("X_val_or.npy")
y_val = np.load("y_val_or.npy")

# Load the different training sets and execute training one by one (finetuning)
# Different number of epochs is used per training based on empirical practice
X_train = np.load("X_train_wise.npy")
estimator.fit( X_train, y_train, validation_data=(X_val, y_val))
X_train = np.load("X_train_dos.npy")
estimator.fit( X_train, y_train, validation_data=(X_val, y_val), nb_epoch=7)
X_train = np.load("X_train_equ.npy")
history = estimator.fit( X_train, y_train, validation_data=(X_val, y_val), nb_epoch=7)






# ## Saving model
# Model is saved in json format so that it can be loaded by the simulator.
# Fragment of code extracted from 
#http://machinelearningmastery.com/save-load-keras-deep-learning-models/

# serialize model to JSON
model_json = estimator.model.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
estimator.model.save_weights("model3.h5")
print("Saved model to disk")


