import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
import model
import trainmodel
EPOCHS = 7
LR = 0.01
opt = SGD(learning_rate= LR)

class Plot:
    def __init__(self):
        pass
    
    def draw_plot(self, count, X, y):
        shapes = ["circle", "square", "triangle", "star"]
        fig, axes =plt.subplots(count//4,4, figsize = (16, count))
        for i, ind in enumerate(np.random.randint(0, X.shape[0] , count)):
            ax = axes[i//4][i%4] 
            ax.imshow(X[ind],cmap = 'gray')
            ax.title.set_text(shapes[np.argmax(y[ind])])
            ax.set_xticks([]) 
            ax.set_yticks([])
        plt.savefig("../plot/predict_plot.png")
        plt.show()
        
    def predict(self):
        model.Model().lenet(numChannels=3, imgRows=64, imgCols=64, numClasses=4, pooling="max", activation="relu")
        model.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])
        pred = model.predict(self.testX)
        plot = Plot()
        plot.draw_plot(8, self.testX, pred)
        
    def plot_accuracy(self):
        H1 = trainmodel.TrainModel.train_with_maxpool()
        H2 = trainmodel.TrainModel.train_with_avgpool()
        plt.figure(figsize=(15,5))
        plt.plot(np.arange(0, EPOCHS), H1.history["acc"], label="Max Pool Train Acc")
        plt.plot(np.arange(0, EPOCHS), H1.history["val_acc"], label="Max Pool Test Acc")
        plt.plot(np.arange(0, EPOCHS), H2.history["acc"], label="Avg Pool Train Acc")
        plt.plot(np.arange(0, EPOCHS), H2.history["val_acc"], label="Avg Pool Test Acc")
        plt.title("Comparing Models Train\Test Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper left")
        plt.savefig("../plot/accuracy.png")
        plt.show()
        
    def plot_loss(self):
        H1 = trainmodel.TrainModel.train_with_maxpool()
        H2 = trainmodel.TrainModel.train_with_avgpool()
        plt.plot(np.arange(0, EPOCHS), H1.history["loss"], label="Max Pool Train Loss")
        plt.plot(np.arange(0, EPOCHS), H1.history["val_loss"], label="Max Pool Test Loss")
        plt.plot(np.arange(0, EPOCHS), H2.history["loss"], label="Avg Pool Train Loss")
        plt.plot(np.arange(0, EPOCHS), H2.history["val_loss"], label="Avg Pool Test Loss")
        plt.title("Comparing Models Train\Test Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="upper left")
        plt.savefig("../plot/loss.png")
        plt.show()
            
    
            
if __name__ == '__main__':
    plot = Plot()
    plot.plot_accuracy()