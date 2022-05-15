from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
class Model:
    @staticmethod
    def lenet(numChannels, imgRows, imgCols, numClasses, pooling= "max", activation= "relu"):
            # initialize the model
            model = Sequential()
            inputShape = (imgRows, imgCols, numChannels)

            # add first set of layers: Conv -> Activation -> Pool
            model.add(Conv2D(filters= 6, kernel_size= 5, input_shape= inputShape))
            model.add(Activation(activation))

            if pooling == "max":
                model.add(MaxPooling2D(pool_size= (2, 2), strides= (2, 2)))
            else:
                model.add(AveragePooling2D(pool_size= (2, 2), strides= (2, 2)))

            # add second set of layers: Conv -> Activation -> Pool
            model.add(Conv2D(filters= 16, kernel_size= 5))
            model.add(Activation(activation))

            if pooling == "avg":
                model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
            else:
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            # Flatten -> FC 120 -> Dropout -> Activation
            model.add(Flatten())
            model.add(Dense(120))
            model.add(Dropout(0.5))
            model.add(Activation(activation))

            # FC 84 -> Dropout -> Activation
            model.add(Dense(84))
            model.add(Dropout(0.5))
            model.add(Activation(activation))

            # FC 4-> Softmax
            model.add(Dense(numClasses))
            model.add(Activation("softmax"))
            #print(model.summary())
            return model
    @staticmethod  
    def alexnet(numChannels, imgRows, imgCols, numClasses):
        inputShape = (imgCols, imgRows, numChannels)
        model = Sequential()
        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=inputShape, kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(Activation('relu'))
        # Pooling 
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(4096, input_shape=(224*224*3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Dense Layer
        model.add(Dense(4))
        model.add(Activation('softmax'))
        """
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(numClasses))
        model.add(Activation('softmax'))
        #print(model.summary())
        """
        return model
    @staticmethod
    def alexnet1():
        model = keras.Sequential([
            layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4, activation='softmax')
            
        ])
        return model
        
        
if __name__ == "__main__":
    Model().alexnet(numChannels = 3, imgRows = 64, imgCols = 64, numClasses = 4)