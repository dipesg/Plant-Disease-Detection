# Import Necessary Packages
import pandas as pd
import numpy as np
import cv2
import logger
import os
from os import listdir
from keras.preprocessing.image import img_to_array, array_to_img

class Preprocessor:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../Processing_Logs/ProcessingLog.txt", 'a+')
        
    def convert_image_to_array(self, image_dir):
        self.log_writer.log(self.file_object, "Entered the convert_image_to_array method of the Preprocessor class.")
        self.image_dir = image_dir
        
        try:
            self.log_writer.log(self.file_object, "Reading images.")
            self.image = cv2.imread(self.image_dir)
            if self.image is not None:
                self.image = cv2.resize(self.image, (256, 256))
                return img_to_array(self.image)
                self.log_writer.log(self.file_object, "Image to array conversion Successful.")
            else:
                return np.array([])
        except Exception as e:
            self.log_writer.log(self.file_object, "Exception occured in convert_image_to_array of the Preprocessor class. Exception message:  "+str(e))
            self.log_writer.log(self.file_object, "Convert image to array Unsuccessful. Exited the convert_image_to_array method of the Preprocessor class")
            raise Exception()
                