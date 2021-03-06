import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import model
import matplotlib.pyplot as plt
import random
import logger
import preprocess
import plot
BS = 4
LR = 0.01
EPOCHS = 7
IMG_SIZE = 64
numClasses = 4
opt = SGD(learning_rate= LR)

from tensorflow.python.framework.config import set_memory_growth
tf.compat.v1.disable_v2_behavior()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class TrainModel:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../logs/trainmodel_log.txt", 'a+')
        self.trainX, self.testX, self.trainY, self.testY = preprocess.Preprocess().split_data()
        self.categories = ["circle", "square", "triangle", "star"]
        self.model1 = model.Model().lenet(numChannels=3, imgRows=64, imgCols=64, numClasses=4, pooling="max", activation="relu")
        self.model1.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])
        self.model3 = model.Model().alexnet(numChannels=3, imgRows=64, imgCols=64, numClasses=4)
        self.model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        

    def train_with_maxpool(self):
            model1 = model.Model().lenet(numChannels=3, imgRows=64, imgCols=64, numClasses=4, pooling="max", activation="relu")
            model1.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])
            H1 = model1.fit(self.trainX, self.trainY, validation_data= (self.testX, self.testY), batch_size= BS, epochs= EPOCHS, verbose=1)
            # Evaluate the train and test data
            scores_train = model1.evaluate(self.trainX, self.trainY, verbose= 1)
            scores_test = model1.evaluate(self.testX, self.testY, verbose= 1)
            
            # save the model to disk
            #filename = '../trainedmodel/model/finalized_model.sav'
            #pickle.dump(self.model1, open(filename, 'wb'))

            print("\nModel with Max Pool Accuracy on Train Data: %.2f%%" % (scores_train[1]*100))
            print("Model with Max Pool Accuracy on Test Data: %.2f%%" % (scores_test[1]*100))
            return H1
            
        
    def train_with_avgpool(self):
        try:
            model2 = model.Model().lenet(numChannels=3, imgRows=64, imgCols=64, numClasses=4, pooling= "average", activation="relu")
            model2.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])
            H2 = model2.fit(self.trainX, self.trainY, validation_data= (self.testX, self.testY), batch_size= BS,
                epochs= EPOCHS, verbose=1)

            # Evaluate the train and test data
            scores_train = self.model2.evaluate(self.trainX, self.trainY, verbose= 1)
            scores_test = self.model2.evaluate(self.testX, self.testY, verbose= 1)

            print("\nModel with Max Pool Accuracy on Train Data: %.2f%%" % (scores_train[1]*100))
            print("Model with Max Pool Accuracy on Test Data: %.2f%%" % (scores_test[1]*100))
            return H2
            
        except Exception as e:
            self.log_writer.log(self.file_object,
                                   'Exception occured in train_with_maxpool method of the TrainModel class. Exception message:  ' + str(e))
    """        
    def train_alexnet(self):
        model3 = Model().alexnet(numChannels=3, imgRows=64, imgCols=64, numClasses=4)
        model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        H1 = model3.fit(self.trainX, self.trainY, validation_data= (self.testX, self.testY), batch_size= BS, epochs= EPOCHS, verbose=1)
        # Evaluate the train and test data
        scores_train = self.model3.evaluate(self.trainX, self.trainY, verbose= 1)
        scores_test = self.model3.evaluate(self.testX, self.testY, verbose= 1)
            
        # save the model to disk
        filename = '../trainedmodel/model/alexnet_model.sav'
        pickle.dump(model3, open(filename, 'wb'))

        print("\nModel with Max Pool Accuracy on Train Data: %.2f%%" % (scores_train[1]*100))
        print("Model with Max Pool Accuracy on Test Data: %.2f%%" % (scores_test[1]*100))
        return H1
    """
            
    def predict(self):
        predict_model = model.Model().lenet(numChannels=3, imgRows=64, imgCols=64, numClasses=4, pooling="max", activation="relu")
        predict_model.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])
        pred = predict_model.predict(self.testX)
        plotter = plot.Plot()
        plotter.draw_plot(8, self.testX, pred)
        
    def plot_accuracy(self):
        H1 = self.train_with_maxpool()
        #H2 = self.train_with_avgpool()
        plt.figure(figsize=(15,5))
        plt.plot(np.arange(0, EPOCHS), H1.history["acc"], label="Max Pool Train Acc")
        plt.plot(np.arange(0, EPOCHS), H1.history["val_acc"], label="Max Pool Test Acc")
        #plt.plot(np.arange(0, EPOCHS), H2.history["acc"], label="Avg Pool Train Acc")
        #plt.plot(np.arange(0, EPOCHS), H2.history["val_acc"], label="Avg Pool Test Acc")
        plt.title("Comparing Models Train\Test Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper left")
        plt.savefig("../plot_fig/accuracymaxpool.png")
        plt.show()
    
    def plot_loss(self):
        H1 = self.train_with_maxpool()
        #H2 = self.train_with_avgpool()
        plt.plot(np.arange(0, EPOCHS), H1.history["loss"], label="Max Pool Train Loss")
        plt.plot(np.arange(0, EPOCHS), H1.history["val_loss"], label="Max Pool Test Loss")
        #plt.plot(np.arange(0, EPOCHS), H2.history["loss"], label="Avg Pool Train Loss")
        #plt.plot(np.arange(0, EPOCHS), H2.history["val_loss"], label="Avg Pool Test Loss")
        plt.title("Comparing Models Train\Test Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="upper left")
        plt.savefig("../plot_fig/loss_avgpool.png")
        plt.show()
        
if __name__ == "__main__":
    trainmodel = TrainModel()
    #trainmodel.train_alexnet()
    #trainmodel.plot_loss()
    trainmodel.predict()
    #trainmodel.plot_accuracy()
    