import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time



def trainCNN(root_dir,classes_num):
    start = time.time()

    #DEV = False
    epochs = 20  # number of iterations with the model
    
    #Defining path for train data and test Data
    train_data_path = f'{root_dir}data/train'  #'./data/train' // path for training data set
    validation_data_path = f'{root_dir}data/test'  #'./data/test' // path for testing data set

    """
    Parameters
    """
    
    #parameters for various fuction 
    img_width, img_height = 150, 150   # setting the lenth and width of each image
    batch_size = 32    # specifies number of images taken in batch
    samples_per_epoch = 1000 # specifies total samples per epochs
    validation_steps = 300  # specifies the for testing
    nb_filters1 = 32  # neurons for the first convulution layer
    nb_filters2 = 64  # neurons for the first convulution layer
    conv1_size = 3
    conv2_size = 2
    pool_size = 2
    #classes_num = 2
    lr = 0.0004

#stated generating a sequention model
    model = Sequential()
    model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))# Convulution layer
    model.add(Activation("relu"))#'relu' is the algorithim used for creating the model
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size))) #pooling layer

    model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

    model.add(Flatten())#fully connected layer for tidng the data
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax')) 

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])  #combile the model that we have created

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) # for image augmentation on data set for training set

    test_datagen = ImageDataGenerator(rescale=1. / 255)  # for image augmentation on data set for testing set

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical') # create the augmented data set for training

    validation_generator = test_datagen.flow_from_directory(
        validation_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical') # create the augmented data set for testing

    """
    Tensorboard log
    """
    log_dir = f'{root_dir}tf-log/'
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    cbks = [tb_cb]

    model.fit_generator(
        train_generator,
        samples_per_epoch=samples_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=cbks,
        validation_steps=validation_steps) # fit the dataset and genetae the model

    target_dir = f'{root_dir}models/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model.save(f'{target_dir}model.h5')
    model.save_weights(f'{target_dir}weights.h5') #  model and weight file is created at defined location

    #Calculate execution time
    end = time.time()
    dur = end-start

    if dur<60:
        print("Execution Time:",dur,"seconds")
    elif dur>60 and dur<3600:
        dur=dur/60
        print("Execution Time:",dur,"minutes")
    else:
        dur=dur/(60*60)
        print("Execution Time:",dur,"hours")


if __name__ == "__main__":
    trainCNN(root_dir='C:/finaltesting/data_set/', classes_num=2)