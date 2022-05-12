from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
class LeNet():
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, pooling= "max", activation= "relu"):
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
            print(model.summary())
            return model
        
if __name__ == "__main__":
    LeNet().build(3, 64, 64, 4, pooling="max", activation="relu")