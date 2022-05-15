import os
import cv2
import logger

class LoadData:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/loaddata_log.txt", 'a+')
        self.PATH = "../images/"
        self.IMG_SIZE = 64
        self.Shapes = ["circle", "square", "triangle", "star"]
        self.Labels = []
        self.Dataset = []
    
    def load_data(self):
        try:
            for shape in self.Shapes:
                self.log_writer.log(self.file_object, 'Loading data...'+ shape)
                print("Getting data for: ", shape)
                #iterate through each file in the folder
                for path in os.listdir(self.PATH + shape):
                    #add the image to the list of images
                    image = cv2.imread(self.PATH + shape + '/' + path)
                    image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
                    self.Dataset.append(image)
                    #add an integer to the labels list 
                    self.Labels.append(self.Shapes.index(shape))

            #print("\nDataset Images size:", len(self.Dataset))
            #print("Image Shape:", self.Dataset[0].shape)
            #print("Labels size:", len(self.Labels))
            
            return self.Dataset, self.Labels
        
        except Exception as e:
            self.log_writer.log(self.file_object,
                                   'Exception occured in load_data method of the LoadData class. Exception message:  ' + str(e))
            raise Exception()
    
if __name__ == "__main__":
    LoadData().load_data()