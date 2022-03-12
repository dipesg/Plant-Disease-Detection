# Doing the necessary import
import numpy as np
import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import os
from os import listdir
from PIL import Image
from sklearn.preprocessing import label_binarize,  LabelBinarizer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical 
import logger
from preprocessing import Preprocessor
import datapath as data 

class trainModel:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../Training_Logs/ModelTrainingLog.txt", 'a+')
        self.Preprocessor = Preprocessor()
        
    def trainingModel(self):
        # Logging the start of training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            """doing the preprocessing"""
            
            self.log_writer.log(self.file_object, 'Doing Preprocessing.')
            # Doing necessary preprocessing
            self.log_writer.log(self.file_object, 'I enter Preprocessing.')
             
            dir = "../dataset"
            root_dir = listdir(dir)
            image_list, label_list = [], []
            all_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
            binary_labels = [0,1,2]
            temp = -1

            # Reading and converting image to numpy array
            for directory in root_dir:
                plant_image_list = listdir(f"{dir}/{directory}")
                temp += 1
                for files in plant_image_list:
                    image_path = f"{dir}/{directory}/{files}"
                    image_list.append(self.Preprocessor.convert_image_to_array(image_path))
                    label_list.append(binary_labels[temp])
            
            # splitting the data into training and test set
            self.log_writer.log(self.file_object, 'Doing train_test_split.')
            x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state = 10)  
            self.log_writer.log(self.file_object, 'Finish train_test_split.')
            
            # Normalizing the dataset
            self.log_writer.log(self.file_object, 'Doing Normalizing.')
            x_train = np.array(x_train, dtype=np.float16) / 225.0
            x_test = np.array(x_test, dtype=np.float16) / 225.0
            x_train = x_train.reshape( -1, 256,256,3)
            x_test = x_test.reshape( -1, 256,256,3)
            
            # Changing to categorical
            self.log_writer.log(self.file_object, 'Changing to categorical.')
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
            
            # Model Building
            self.log_writer.log(self.file_object, 'Model Building.')
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding="same",input_shape=(256,256,3), activation="relu"))
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(8, activation="relu"))
            model.add(Dense(3, activation="softmax"))
            model.summary()
            
            
            # Compiling Modell
            self.log_writer.log(self.file_object, 'Compiling Model.')
            model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0001), metrics=['accuracy'])
            
            # Again splitting the training dataset into training and validation datasets
            self.log_writer.log(self.file_object, 'Again doing train_test_split.')
            x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size = 0.2)
            
            # Training the model
            epochs = 50
            batch_size = 20
            self.log_writer.log(self.file_object, 'Fitting a model.')
            history = model.fit(x_tr, y_tr, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val), shuffle=True)
            
            # Saving Model
            self.log_writer.log(self.file_object, 'Saving Model.')
            model.save("../dataset/disease.h5")
            # Serialize model to json
            json_model = model.to_json()
            # save the model architechture to JSON file
            with open("../dataset/disease.json", "w") as json_file:
                json_file.write(json_model)
                
            #saving the weights of the model
            model.save_weights("../dataset/disease_weights.h5")
            
            #Plot the training history
            self.log_writer.log(self.file_object, 'Plotting the training history.')
            plt.figure(figsize=(12, 5))
            plt.plot(history.history['accuracy'], color='r')
            plt.plot(history.history['val_accuracy'], color='b')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            plt.legend(['train', 'val'])
            plt.show()
            plt.savefig("../plot/plot.png")   # save the figure to file
            
            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()
            
        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, "Exception occured in tainingModel of trainModel class. Exception message:  "+str(e))
            self.log_writer.log(self.file_object, "trainingModel Unsuccessful. Exited the trainingModel method of the trainModel class")
            raise Exception()