import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Average
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import Loss
class MCNN(Model):
    def __init__(self):
        super(MCNN, self).__init__()
        self.branch0 = Sequential([
            Conv2D(filters=12, kernel_size=13, activation="relu", padding="same"),
            MaxPooling2D(pool_size=2), 
            Conv2D(filters=24, kernel_size=11, activation="relu", padding="same"),
            MaxPooling2D(pool_size=2), 
            Conv2D(filters=12, kernel_size=11, activation="relu", padding="same"),
            #MaxPooling2D(pool_size=2), 
            Conv2D(filters=6, kernel_size=11, activation="relu", padding="same"),
        ])
        self.branch1 = Sequential([
            Conv2D(filters=16, kernel_size=9, activation="relu", padding="same"),
            MaxPooling2D(pool_size=2), 
            Conv2D(filters=32, kernel_size=7, activation="relu", padding="same"),
            MaxPooling2D(pool_size=2), 
            Conv2D(filters=16, kernel_size=7, activation="relu", padding="same"),
            #MaxPooling2D(pool_size=2), 
            Conv2D(filters=8, kernel_size=7, activation="relu", padding="same"),
        ])
        
        self.branch2 = Sequential([
            Conv2D(filters=20, kernel_size=7, activation="relu", padding="same"),
            MaxPooling2D(pool_size=2), 
            Conv2D(filters=40, kernel_size=5, activation="relu", padding="same"),
            MaxPooling2D(pool_size=2), 
            Conv2D(filters=20, kernel_size=5, activation="relu", padding="same"),
            #MaxPooling2D(pool_size=2), 
            Conv2D(filters=10, kernel_size=5, activation="relu", padding="same"),
        ])
        
        self.branch3 = Sequential([
            Conv2D(filters=24, kernel_size=5, activation="relu", padding="same"),
            MaxPooling2D(pool_size=2), 
            Conv2D(filters=48, kernel_size=3, activation="relu", padding="same"),
            MaxPooling2D(pool_size=2), 
            Conv2D(filters=24, kernel_size=3, activation="relu", padding="same"),
            #MaxPooling2D(pool_size=2), 
            Conv2D(filters=12, kernel_size=3, activation="relu", padding="same"),
        ])
        
        self.branch4 = Sequential([
            Conv2D(filters=28, kernel_size=3, activation="relu", padding="same"),
            MaxPooling2D(pool_size=2), 
            Conv2D(filters=56, kernel_size=1, activation="relu", padding="same"),
            MaxPooling2D(pool_size=2), 
            Conv2D(filters=28, kernel_size=1, activation="relu", padding="same"),
            #MaxPooling2D(pool_size=2), 
            Conv2D(filters=14, kernel_size=1, activation="relu", padding="same"),
        ])
        
        self.fuse = Sequential([
            Conv2D(filters=1, kernel_size=1, padding="valid")
        ])
        
    def call(self, inputs):
        hidden0, hidden1, hidden2, hidden3, hidden4 = self.branch0(inputs), self.branch1(inputs), self.branch2(inputs), self.branch3(inputs), self.branch4(inputs)
        cat = tf.concat([hidden0, hidden1, hidden2, hidden3, hidden4], axis=3)
        output = self.fuse(cat)
        return output
    
class MCNNLoss(Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y_true, y_pred):
        count_loss = abs(tf.math.reduce_sum(y_true) - tf.math.reduce_sum(y_pred))
        return count_loss
