import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import logger
from data import LoadData
class Preprocess:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/preprocess_log.txt", 'a+')   
        
    def normalize_data(self):
        try:
            self.log_writer.log(self.file_object, 'Normalizing Images.')
            dataset, labels = LoadData().load_data()
            Dataset = np.array(dataset)
            Dataset = Dataset.astype("float32") / 255.0

            self.log_writer.log(self.file_object, 'One hot encoding labels.')
            Labels = np.array(labels)
            Labels = to_categorical(Labels)
            return Dataset, Labels
        
        except Exception as e:
            self.log_writer.log(self.file_object,
                                   'Exception occured in normalize_data method of the Preprocess class. Exception message:  ' + str(e))
            raise Exception()
        
    def split_data(self):
        try:
            dataset, labels = self.normalize_data()
            X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
            print(X_test.shape[0])
            return X_train, X_test, y_train, y_test
         
        
        except Exception as e:
            self.log_writer.log(self.file_object,
                                   'Exception occured in split_data method of the Preprocess class. Exception message:  ' + str(e))
        
if __name__ == "__main__":
    Preprocess().split_data()