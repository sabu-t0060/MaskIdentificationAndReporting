import sys
import os
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model




def fetch_model(rootpath):
    #Define Path
    model_path = f'{rootpath}models/model.h5'#'./models/model.h5'
    model_weights_path = f'{rootpath}models/weights.h5'#'./models/weights.h5'
    #test_path = '../data_set/data/alien_test'  #'./data/alien_test'

    #Load the pre-trained models
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    return model

#Prediction Function
def predict(file,model):
    # Define image parameters
    img_width, img_height = 150, 150
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    #print(result)
    answer = np.argmax(result)
    print(f'{file} --{answer}')
    return answer


if __name__ == "__main__":
    #trainCNN(root_dir='../data_set/', classes_num=2)

    #model = fetch_model('../data_set/')
    #filename = './mask/camera_0.jpg'
    #result = predict(filename, model)
    print('Done')