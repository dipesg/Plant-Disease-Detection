import matplotlib.pyplot as plt
from matplotlib.image import imread
import logger
import random
import os

class Visualize:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../Visualization_Logs/VisualizationLog.txt", 'a+')
    def plot(self, path):
        self.path = path
        self.log_writer.log(self.file_object, 'Entered the plot method of the Visualize class')
        plt.figure(figsize=(12, 12))
        
        for i in range(1, 17):
            plt.subplot(4, 4, i)
            plt.tight_layout()
            self.rand_img = imread(self.path +'/'+ random.choice(sorted(os.listdir(self.path))))
            plt.imshow(self.rand_img)
            plt.xlabel(self.rand_img.shape[1], fontsize = 10) # Width of image
            plt.ylabel(self.rand_img.shape[0], fontsize = 10) # height of image
        plt.show()
        plt.savefig("../plot/fig1.png")

if __name__ == "__main__":
    log_writer = logger.App_Logger()
    file_object = open("../Visualization_Logs/VisualizationLog.txt", 'a+')
    log_writer.log(file_object, 'Entered the Main part.')
    path = "../dataset/Potato_Early_blight" # You can give path to any data folder and can see the figure.
    Visualize().plot(path)