#downloading data from keggle
#### Data downloading ....
!pip install kaggle

from google.colab import files

files.upload()

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d alxmamaev/flowers-recognition

import zipfile
zip_ref = zipfile.ZipFile("flowers-recognition.zip")
zip_ref.extractall("files")
zip_ref.close()

# importing modules
import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense , Conv2D , Dropout , MaxPooling2D , Flatten, Activation , BatchNormalization
from tensorflow.keras.models import Sequential , Model
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt


#print(len(os.listdir('/content/files/flowers/daisy')))
#print(len(os.listdir('/content/files/flowers/dandelion')))
#print(len(os.listdir('/content/files/flowers/rose')))
#print(len(os.listdir('/content/files/flowers/sunflower')))
#print(len(os.listdir('/content/files/flowers/tulip')))

#######Directories were created for extracting images   ######
# Dataset
# Dataset/training
# Dataset/testing
# each of above contained 5 directories of 5 flowers - daisy,dandelion,sunflower,rose,tulip

os.mkdir("/content/flower_dataset")
os.mkdir("/content/flower_dataset/training")
os.mkdir("/content/flower_dataset/testing")
os.mkdir("/content/flower_dataset/training/daisy")
os.mkdir("/content/flower_dataset/training/dandelion")
os.mkdir("/content/flower_dataset/training/rose")
os.mkdir("/content/flower_dataset/training/sunflower")
os.mkdir("/content/flower_dataset/training/tulip")
os.mkdir("/content/flower_dataset/testing/daisy")
os.mkdir("/content/flower_dataset/testing/dandelion")
os.mkdir("/content/flower_dataset/testing/rose")
os.mkdir("/content/flower_dataset/testing/sunflower")
os.mkdir("/content/flower_dataset/testing/tulip")



# Splitting data in 90/10 ratio
split_size = .9
#function to split total data into training and testing set

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    # SOURCE is path in which images of certaing flower was extracred
    # TRAINING is path where trainig images will be stored
    # TESTING is path where validation images will be stored
    all_images = os.listdir(SOURCE)
    #print(type(all_images)) type list
    #Shuffling is imoortant before splitting
    random.shuffle(all_images)
    splitting_index = round(SPLIT_SIZE*len(all_images))
    # Selecting images from 0 to 90% of total images of certtaing flower for training_purpose
    train_images = all_images[:splitting_index]
    # rest for validation
    test_images = all_images[splitting_index:]

    #Putting every image to desired path
    for img in train_images:
        shutil.copy(os.path.join(SOURCE,img),TRAINING)
    for img in test_images:
        shutil.copy(os.path.join(SOURCE,img),TESTING)


# path to individual directories#
DAISY_SOURCE_DIR = "/content/files/flowers/daisy"
DANDELION_SOURCE_DIR = "/content/files/flowers/dandelion"
ROSE_SOURCE_DIR = "/content/files/flowers/rose"
SUNFLOWER_SOURCE_DIR = "/content/files/flowers/sunflower"
TULIP_SOURCE_DIR = "/content/files/flowers/tulip"
TRAINING_DAISY_DIR = "/content/flower_dataset/training/daisy"
TRAINING_DANDELION_DIR = "/content/flower_dataset/training/dandelion"
TRAINING_ROSE_DIR = "/content/flower_dataset/training/rose"
TRAINING_SUNFLOWER_DIR = "/content/flower_dataset/training/sunflower"
TRAINING_TULIP_DIR = "/content/flower_dataset/training/tulip"
TESTING_DAISY_DIR = "/content/flower_dataset/testing/daisy"
TESTING_DANDELION_DIR = "/content/flower_dataset/testing/dandelion"
TESTING_ROSE_DIR = "/content/flower_dataset/testing/rose"
TESTING_SUNFLOWER_DIR = "/content/flower_dataset/testing/sunflower"
TESTING_TULIP_DIR = "/content/flower_dataset/testing/tulip"


# calling the above function that takes path of directories , splits available data into training and testing set and
# stores it in that flower's training and testing directory
#

split_data(DAISY_SOURCE_DIR , TRAINING_DAISY_DIR ,TESTING_DAISY_DIR , split_size)
split_data(DANDELION_SOURCE_DIR , TRAINING_DANDELION_DIR ,TESTING_DANDELION_DIR , split_size)
split_data(ROSE_SOURCE_DIR , TRAINING_ROSE_DIR ,TESTING_ROSE_DIR , split_size)
split_data(SUNFLOWER_SOURCE_DIR , TRAINING_SUNFLOWER_DIR ,TESTING_SUNFLOWER_DIR , split_size)
split_data(TULIP_SOURCE_DIR , TRAINING_TULIP_DIR ,TESTING_TULIP_DIR , split_size)



#print(len(os.listdir('/content/flower_dataset/training/daisy')))
#print(len(os.listdir('/content/flower_dataset/training/dandelion')))
#print(len(os.listdir('/content/flower_dataset/training/rose')))
#print(len(os.listdir('/content/flower_dataset/training/sunflower')))
#print(len(os.listdir('/content/flower_dataset/training/tulip')))
#print(len(os.listdir('/content/flower_dataset/testing/daisy')))
#print(len(os.listdir('/content/flower_dataset/testing/dandelion')))
#print(len(os.listdir('/content/flower_dataset/testing/rose')))
#print(len(os.listdir('/content/flower_dataset/testing/sunflower')))
#print(len(os.listdir('/content/flower_dataset/testing/tulip')))
## training set for daisy -692
## training set for dandelion -950
## training set for rose -706
## training set for sunflower -661
## training set for tulip -886
## validation set for daisy -377
## validation set for dandelion -105
##validation set for rose -78
## validation set for sunflower -73
## validation set for tulip -98


# Extracting data from training and testing directories
# Image preprocessing
# image size converted to 180x180
from tensorflow.keras.preprocessing.image import ImageDataGenerator
TRAINING_DIR = '/content/flower_dataset/training'
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.0,
                                                               rotation_range=30,
                                                                width_shift_range=0.3,
                                                              height_shift_range=0.3,
                                                              shear_range=0.4,
                                                              zoom_range=0.1,
                                                              horizontal_flip=True)
# Train-generator contains training example with training labels . It is an iterator of type-
# tensorflow.python.keras.preprocessing.image.DirectoryIterator
train_generator =  train_datagen.flow_from_directory(TRAINING_DIR,
                                                      target_size=(180, 180),
                                                      batch_size=128,
                                                     class_mode='categorical')
TESTING_DIR = '/content/flower_dataset/testing'
testing_datagen = ImageDataGenerator(rescale=1./255.0)
# Testing-generator contains validation examples with validation labels
testing_generator =  testing_datagen.flow_from_directory(TESTING_DIR,
                                                      target_size=(180, 180),
                                                      batch_size=128,
                                                      class_mode='categorical')


##### Found 3892 images belonging to 5 classes -- for training
##### Found 431 images belonging to 5 classes -- for validation




##### Trainnig the model ######
###                         ###
#                             #
model = Sequential()
model.add(Conv2D(16,(3,3),strides=(1,1),activation='relu',input_shape=(180,180,3)))
model.add(MaxPooling2D((3,3),strides=1,padding='valid'))
model.add(Conv2D(32,(3,3),strides=(1,1),activation='relu'))
model.add(MaxPooling2D((3,3),strides=1,padding='valid'))
model.add(Conv2D(64,(3,3),strides=(2,2),activation='relu'))
model.add(MaxPooling2D((3,3),strides=2,padding='valid'))
model.add(Conv2D(128,(5,5),strides=(2,2),activation='relu'))
model.add(MaxPooling2D((5,5),strides=2,padding='valid'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu',kernel_regularizer='l2'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.summary()
#print("input shape of model =",model.input_shape)

# This model is trained over 2,400,965 parameters for 80 iterations
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# Binary crossentropy gave better results with adam optimizer in this project

history = model.fit(train_generator,epochs=80,batch_size=128,validation_data=testing_generator)
## After training model for 80 iterations ,
## trainnig loss: 0.1670 - accuracy: 0.8410  -  val_loss: 0.1776 - val_accuracy: 0.8329


##plotting the model graph for loss and accuracy with iterations ...
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

#the graph of accuracy showed saturation near 85% accuracy training....

model.evaluate(testing_generator)











daisy1 = cv2.imread("/content/daisy1.jpg")
daisy2 = cv2.imread("/content/daisy2.jpg")
daisy3 = cv2.imread("/content/daisy3.jpg")
daisy4 = cv2.imread("/content/daisy4.jpg")
daisy5 = cv2.imread("/content/daisy5.jpg")
dandelion1 = cv2.imread("/content/dandelion1.jpg")
dandelion2 = cv2.imread("/content/dandelion2.jpg")
dandelion3 = cv2.imread("/content/dandelion3.jpg")
dandelion4 = cv2.imread("/content/dandelion4.jpg")
dandelion5 = cv2.imread("/content/dandelion5.jpg")
rose1 = cv2.imread("/content/Red_rose1.jpg")
rose2 = cv2.imread("/content/sample_data/testing/rose2.png")
rose3 = cv2.imread("/content/sample_data/testing/rose3.jpg")
rose4 = cv2.imread("/content/sample_data/testing/rose4.jpg")
sunflower1 = cv2.imread("/content/sunflower1.jpg")
sunflower2 = cv2.imread("/content/sunflower2.jpg")
sunflower3 = cv2.imread("/content/sunflower3.jpg")
sunflower4 = cv2.imread("/content/sunflower4.jpg")
sunflower5 = cv2.imread("/content/sunflower5.jpg")
tulip2 = cv2.imread("/content/tulip2.jpg")
tulip3 = cv2.imread("/content/tulip3.jpg")
tulip4 = cv2.imread("/content/tulip4.jpg")
tulip5 = cv2.imread("/content/tulip5.jpg")



c = [daisy1,daisy2,daisy3,daisy4,daisy5,dandelion1,dandelion2,dandelion3,dandelion4,dandelion5,rose1,rose2,sunflower1,sunflower2,sunflower3,sunflower4,sunflower5,tulip2,tulip3,tulip4,tulip5]
flower = {0:'daisy',1:'dandelion',2:'rose',3:'sunflower',4:'tulip'}




for i in c:
    img = i
    plt.figure()
    plt.imshow(img)

    img = cv2.resize(i, (180, 180))
    img = np.reshape(img, (1, 180, 180, 3))
    print(flower[np.argmax(model.predict(img))])




import numpy as np

from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
    # predicting images
    path = '/content/' + fn
    img = image.load_img(path, target_size=(180, 180))
    plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images)

    print(flower[np.argmax(classes)])