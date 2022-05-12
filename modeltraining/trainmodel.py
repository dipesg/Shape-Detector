import numpy as np
import pickle
from keras.optimizers import SGD
from model import LeNet
import seaborn as sns
import matplotlib.pyplot as plt
import random
import logger
from preprocess import Preprocess
BS = 16
LR = 0.01
EPOCHS = 10
IMG_SIZE = 64
numClasses = 4
opt = SGD(lr= LR)

class TrainModel:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/trainmodel_log.txt", 'a+')
        self.trainX, self.testX, self.trainY, self.testY = Preprocess().split_data()
        self.categories = ["circle", "square", "triangle", "star"]
        self.model1 = LeNet().build(numChannels=3, imgRows=64, imgCols=64, numClasses=4, pooling="max", activation="relu")
        self.model1.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])
        
    def train_with_maxpool(self):
            self.model1 = LeNet().build(numChannels=3, imgRows=64, imgCols=64, numClasses=4, pooling="max", activation="relu")
            self.model1.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])
            self.H1 = self.model1.fit(self.trainX, self.trainY, validation_data= (self.testX, self.testY), batch_size= BS, epochs= EPOCHS, verbose=1)
            # Evaluate the train and test data
            scores_train = self.model1.evaluate(self.trainX, self.trainY, verbose= 1)
            scores_test = self.model1.evaluate(self.testX, self.testY, verbose= 1)
            
            # save the model to disk
            filename = '../trainedmodel/model/finalized_model.sav'
            pickle.dump(self.model1, open(filename, 'wb'))

            print("\nModel with Max Pool Accuracy on Train Data: %.2f%%" % (scores_train[1]*100))
            print("Model with Max Pool Accuracy on Test Data: %.2f%%" % (scores_test[1]*100))
            
        
    def train_with_avgpool(self):
        try:
            self.model2 = LeNet.build(numChannels=3, imgRows=64, imgCols=64, numClasses=4, pooling= "average", activation="relu")
            self.model2.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])
            self.H2 = self.model2.fit(self.trainX, self.trainY, validation_data= (self.testX, self.testY), batch_size= BS,
                epochs= EPOCHS, verbose=1)

            # Evaluate the train and test data
            scores_train = self.model2.evaluate(self.trainX, self.trainY, verbose= 1)
            scores_test = self.model2.evaluate(self.testX, self.testY, verbose= 1)

            print("\nModel with Max Pool Accuracy on Train Data: %.2f%%" % (scores_train[1]*100))
            print("Model with Max Pool Accuracy on Test Data: %.2f%%" % (scores_test[1]*100))
            
        except Exception as e:
            self.log_writer.log(self.file_object,
                                   'Exception occured in train_with_maxpool method of the TrainModel class. Exception message:  ' + str(e))
            
    def predict(self):
        predictions = self.model1.predict(self.testX)
        fig = plt.figure(figsize=(20,15))
        gs = fig.add_gridspec(4, 4)
        for line in range(0, 3):
            for row in range(0, 3):
                num_image = random.randint(0, self.testX.shape[0])
                ax = fig.add_subplot(gs[line, row])
                ax.axis('off');
                ax.set_title("Predicted: " + self.categories[list(np.round(predictions[num_image]))])
                ax.imshow(self.testX[num_image]);
        fig.suptitle("Predicted label for the displayed shapes", fontsize=25, x=0.42);
        
    def plot_accuracy(self):
        plt.figure(figsize=(15,5))
        plt.plot(np.arange(0, EPOCHS), self.H1.history["acc"], label="Max Pool Train Acc")
        plt.plot(np.arange(0, EPOCHS), self.H1.history["val_acc"], label="Max Pool Test Acc")
        plt.plot(np.arange(0, EPOCHS), self.H2.history["acc"], label="Avg Pool Train Acc")
        plt.plot(np.arange(0, EPOCHS), self.H2.history["val_acc"], label="Avg Pool Test Acc")
        plt.title("Comparing Models Train\Test Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper left")
        plt.savefig("../plot/accuracy.png")
        
if __name__ == "__main__":
    trainmodel = TrainModel()
    #trainmodel.train_with_avgpool()
    #trainmodel.plot_accuracy()
    trainmodel.predict()
    